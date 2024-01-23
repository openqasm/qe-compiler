# (C) Copyright IBM 2023.
#
# This code is part of Qiskit.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file defines the compile interface for the qss_compiler
package.

Developer notes:
This further decouples the python package from the _execution_ of
compile.

To update the user interface, update `compile` with the new
interface, and update the C++ function `py_compile` bound to
`_compile` in `lib.cpp`.


Why fork a child process for compilation?
------------------------------------------

LLVM is designed to compile one program per execution. However, we
would like a python user to be able to reliably call `compile` as
many times as they like with different inputs with some of the calls
possibly occurring in parallel. Additionally, it would not be ideal if
we try to compile some input and our python interpreter crashes because
the input was malformed, and our code raised an error.

To handle this, we call `_compile` (which is defined in the shared
library `py_qssc` as described by `python_lib/lib.cpp`) within a
child process. This guarantees the calling process won't be killed,
even if the call to `_compile` results in a segmentation fault.

For forking child processes and communicating with them, we use the
Python standard library's multiprocessing package. In particular, we use
the Process class to start a child process with a python interpreter and
pass control to our backend function. We use multiprocessing's Pipe to
pass input and output data to and from the compiling process. Process
and Pipe take care of serializing python objects and passing them across
process boundaries with pipes.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources as importlib_resources
import multiprocessing as mp
from multiprocessing import connection
from os import environ as os_environ
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from . import exceptions
from .py_qssc import _compile_with_args, Diagnostic

# use the forkserver context to create a server process
# for forking new compiler processes
mp_ctx = mp.get_context("forkserver")


class InputType(Enum):
    """Enumeration of input types recognized by the compiler."""

    NONE = "none"
    QASM3 = "qasm"
    MLIR = "mlir"
    BYTECODE = "bytecode"

    def __str__(self):
        return self.value


class OutputType(Enum):
    """Enumeration of output types supported by the compiler"""

    QEM = "qem"
    MLIR = "mlir"
    BYTECODE = "bytecode"

    def __str__(self):
        return self.value


@dataclass
class CompileOptions:
    """Options to the compiler backend.

    Most correspond directly to qss-compiler CLI arguments.
    """

    """Input source type."""
    input_type: InputType = InputType.QASM3
    """Output source type."""
    output_type: OutputType = OutputType.QEM
    """Output file, if not supplied raw bytes will be returned."""
    output_file: Union[str, None] = None
    """Hardware target to select."""
    target: Optional[str] = None
    """Hardware configuration location."""
    config_path: Union[str, None] = None
    """Number of shots to run each circuit for."""
    num_shots: Optional[int] = None
    """Repetition delay between shots in seconds."""
    shot_delay: Optional[float] = None
    """Optional list of extra arguments to pass to the compiler.

    Individual arguments must be separate arguments in the list.

    Example: ["--print-ir-after-all", "-other-arg"]

    Certain extra arguments may clash with other :class:`CompileOptions`
    resulting in args being set twice, eg., `num_shots` and `--num-shots`.
    This will result in an error.
    """
    extra_args: List[str] = field(default_factory=list)
    """Optional callback for processing diagnostic messages from the compiler."""
    on_diagnostic: Optional[Callable[[Diagnostic], Any]] = None

    def prepare_compiler_option_args(self) -> List[str]:
        """Prepare the compiler option arguments from this dataclass."""
        args = [
            "qss-compiler",
            f"-X={str(self.input_type)}",
            f"--emit={str(self.output_type)}",
        ]
        if self.output_file:
            args.append("-o")
            args.append(str(self.output_file))

        if self.target:
            args.append(f"--target={str(self.target)}")

        if self.config_path:
            args.append(f"--config={str(self.config_path)}")

        if self.num_shots:
            args.append(f"--num-shots={self.num_shots}")

        if self.shot_delay:
            args.append(f"--shot-delay={self.shot_delay*1e6}us")

        args.extend(self.extra_args)
        return args


@dataclass
class _CompilerExecution:
    """Internal compiler execution dataclass."""

    input: Optional[Union[str, bytes]] = None
    input_file: Optional[Union[str, Path]] = None
    options: CompileOptions = field(default_factory=CompileOptions)

    def prepare_compiler_args(self) -> List[str]:
        args = self.options.prepare_compiler_option_args()

        if self.input_file is not None:
            args.append(str(self.input_file))
        elif self.input is not None:
            if self.options.input_type == InputType.BYTECODE:
                args.append(self.input.hex())
            else:
                args.append(str(self.input))
        else:
            raise exceptions.QSSCompilerNoInputError(
                "Neither input file nor input string provided."
            )

        return args


@dataclass
class _CompilerStatus:
    """Internal compiler result status dataclass."""

    success: bool


def _compile_child_backend(
    execution: _CompilerExecution,
    on_diagnostic: Callable[[Diagnostic], Any],
) -> Tuple[_CompilerStatus, Union[bytes, None]]:
    # TODO: want a corresponding C++ interface to avoid overhead

    options = execution.options
    args = execution.prepare_compiler_args()
    output_as_return = False if options.output_file else True

    # The qss-compiler expects the path to static resources in the environment
    # variable QSSC_RESOURCES. In the python package, those resources are
    # bundled under the directory resources/. Since python's functions for
    # looking up resources only treat files as resources, use the generated
    # python source _version.py to look up the path to the python package.
    with importlib_resources.path("qss_compiler", "_version.py") as version_py_path:
        resources_path = version_py_path.parent / "resources"
        os_environ["QSSC_RESOURCES"] = str(resources_path)
        success, output = _compile_with_args(args, output_as_return, on_diagnostic)

    status = _CompilerStatus(success)
    if output_as_return:
        return status, output
    else:
        return status, None


def _compile_child_runner(conn: connection.Connection) -> None:
    execution = conn.recv()

    def on_diagnostic(diag):
        conn.send(diag)

    status, output = _compile_child_backend(execution, on_diagnostic)
    conn.send(status)
    if output is not None:
        conn.send_bytes(output)


def _do_compile(
    execution: _CompilerExecution, return_diagnostics: bool = False
) -> Union[bytes, str, None]:
    assert (
        execution.input_file is not None or execution.input is not None
    ), "one of the compile options input_file or input must be set"

    options = execution.options

    parent_side, child_side = mp_ctx.Pipe(duplex=True)

    try:
        childproc = mp_ctx.Process(target=_compile_child_runner, args=(child_side,))
        childproc.start()

        parent_side.send(execution)

        # we must handle the case when the child process exits without
        # delivering a status and (if requested) a result (e.g., when
        # crashing). for that purpose, close the pipe's send side in the parent
        # so that blocking receives will be interrupted if the child process
        # exits and closes its end of the pipe.
        child_side.close()

        success = False
        # when no callback was provided, collect diagnostics and return in case of error
        diagnostics = []
        try:
            while True:
                received = parent_side.recv()

                if isinstance(received, Diagnostic):
                    if options.on_diagnostic:
                        options.on_diagnostic(received)
                    else:
                        diagnostics.append(received)
                elif isinstance(received, _CompilerStatus):
                    success = received.success
                    break
                else:
                    childproc.kill()
                    childproc.join()
                    raise exceptions.QSSCompilerCommunicationFailure(
                        "The compile process delivered an unexpected object instead of status or "
                        "diagnostic information. This points to inconsistencies in the Python "
                        "interface code between the calling process and the compile process.",
                        return_diagnostics=return_diagnostics,
                    )

            if options.output_file is None:
                # return compilation result via IPC instead of in a file.
                output = parent_side.recv_bytes()
            else:
                output = None
        except EOFError:
            # make sure that child process terminates
            childproc.kill()
            childproc.join()
            raise exceptions.QSSCompilerEOFFailure(
                "Compile process exited before delivering output.",
                diagnostics,
                return_diagnostics=return_diagnostics,
            )

        childproc.join()
        if childproc.exitcode != 0:
            raise exceptions.QSSCompilerNonZeroStatus(
                (
                    "Compile process exited with non-zero status "
                    + str(childproc.exitcode)
                    + (" yet appears  still alive" if childproc.is_alive() else "")
                ),
                diagnostics,
                return_diagnostics=return_diagnostics,
            )

        if not success:
            raise exceptions.QSSCompilationFailure(
                "Failure during compilation",
                diagnostics,
                return_diagnostics=return_diagnostics,
            )

    except mp.ProcessError as e:
        raise exceptions.QSSCompilerError(
            "It's likely that you've hit a bug in the QSS Compiler. Please "
            "submit an issue to the team with relevant information "
            "(https://github.com/Qiskit/qss-compiler/issues):\n"
            f"{e}",
            return_diagnostics=return_diagnostics,
        )

    if options.output_file is None:
        # return compilation result
        if options.output_type == OutputType.MLIR:
            return output.decode("utf8")
        return output


def _prepare_compile_options(
    compile_options: Optional[CompileOptions] = None, **kwargs
) -> CompileOptions:
    if compile_options is None:
        compile_options = CompileOptions(**kwargs)
    return compile_options


def _stringify_path(p):
    return str(p) if isinstance(p, Path) else p


def compile_file(
    input_file: Union[Path, str],
    return_diagnostics: bool = False,
    compile_options: Optional[CompileOptions] = None,
    **kwargs,
) -> Union[bytes, str, None]:
    """! Compile a file to the specified output type using the given target.

    Produces output in a file (if parameter output_file is provided) or returns
    the compiler output as byte sequence or string, depending on the requested
    output format.

    Args:
        input_file: Path to the input file to compile.
        return_diagnostics: diagnostics visibility flag
        compile_options: Optional :class:`CompileOptions` dataclass.
        kwargs: Keywords corresponding to :class:`CompileOptions`. Ignored if `compile_options`
            is provided directly.

    Returns: Produces output in a file (if parameter output_file is provided) or returns
        the compiler output as byte sequence or string, depending on the requested
        output format.
    """
    input_file = _stringify_path(input_file)
    compile_options = _prepare_compile_options(compile_options, **kwargs)
    execution = _CompilerExecution(input_file=input_file, options=compile_options)
    return _do_compile(execution, return_diagnostics)


async def compile_file_async(
    input_file: Union[Path, str],
    return_diagnostics: bool = False,
    compile_options: Optional[CompileOptions] = None,
    **kwargs,
) -> Union[bytes, str, None]:
    """Compile the given input file to the specified output type using the
    given target in an async context, and avoid blocking the event loop.

    Functionally, this function behaves like compile_str and accepts the same set of parameters.

    Args:
        input_file: Path to the input file to compile.
        return_diagnostics: diagnostics visibility flag
        compile_options: Optional :class:`CompileOptions` dataclass.
        kwargs: Keywords corresponding to :class:`CompileOptions`. Ignored if `compile_options`
            is provided directly.

    Returns: Produces output in a file (if parameter output_file is provided) or returns
        the compiler output as byte sequence or string, depending on the requested
        output format.

    """
    compile_options = _prepare_compile_options(compile_options, **kwargs)
    execution = _CompilerExecution(input_file=input_file, options=compile_options)
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _do_compile, execution, return_diagnostics)
    # As an alternative, ProcessPoolExecutor has somewhat higher overhead yet
    # reduces complexity of integration by not requiring the preparatory call
    # to set_start_method.


def compile_str(
    input: Union[str, bytes],
    return_diagnostics: bool = False,
    compile_options: Optional[CompileOptions] = None,
    **kwargs,
) -> Union[bytes, str, None]:
    """Compile the given input program to the specified output type using the
    given target.

    Produces output in a file (if parameter output_file is provided) or returns
    the compiler output as byte sequence or string, depending on the requested
    output format.

    Args:
        input: input to compile as string or bytes (e.q., an OpenQASM3 program).
        return_diagnostics: diagnostics visibility flag
        compile_options: Optional :class:`CompileOptions` dataclass.
        kwargs: Keywords corresponding to :class:`CompileOptions`. Ignored if `compile_options`
            is provided directly.

    Returns: Produces output in a file (if parameter output_file is provided) or returns
        the compiler output as byte sequence or string, depending on the requested
        output format.
    """
    compile_options = _prepare_compile_options(compile_options, **kwargs)
    execution = _CompilerExecution(input=input, options=compile_options)
    return _do_compile(execution, return_diagnostics=return_diagnostics)


async def compile_str_async(
    input: Union[str, bytes],
    return_diagnostics: bool = False,
    compile_options: Optional[CompileOptions] = None,
    **kwargs,
) -> Union[bytes, str, None]:
    """Compile the given input program to the specified output type using the
    given target in an async context, and avoid blocking the event loop.

    Functionally, this function behaves like compile_str and accepts the same set of parameters.

    Args:
        input: input to compile as string or bytes (e.q., an OpenQASM3 program).
        return_diagnostics: diagnostics visibility flag
        compile_options: Optional :class:`CompileOptions` dataclass.
        kwargs: Keywords corresponding to :class:`CompileOptions`. Ignored if `compile_options`
            is provided directly.

    Returns: Produces output in a file (if parameter output_file is provided) or returns
        the compiler output as byte sequence or string, depending on the requested
        output format.

    """
    compile_options = _prepare_compile_options(compile_options, **kwargs)
    execution = _CompilerExecution(input=input, options=compile_options)
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, _do_compile, execution, return_diagnostics)
    # As an alternative, ProcessPoolExecutor has somewhat higher overhead yet
    # reduces complexity of integration by not requiring the preparatory call
    # to set_start_method.

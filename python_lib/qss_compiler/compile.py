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

To update the user interface, update the Python functions
with the new interface, and update the C++ functions in
lib.cpp to call the C++ compiler via Pybind.


Why fork a child process for compilation?
------------------------------------------

LLVM is designed to compile one program per execution. However, we
would like a python user to be able to reliably call `compile_str/file` as
many times as they like with different inputs with some of the calls
possibly occurring in parallel. Additionally, it would not be ideal if
we try to compile some input and our python interpreter crashes because
the input was malformed, and our code raised an error.

To handle this, we call `compile_file/bytes` (which is defined in the shared
library `py_qssc` as described by `python_lib/lib.cpp`) within a
child process. This guarantees the calling process won't be killed,
even if the call results in a segmentation fault.

For forking child processes and communicating with them, we use the
Python standard library's multiprocessing package. In particular, we use
the Process class to start a child process with a python interpreter and
pass control to our backend function. We use multiprocessing's Pipe to
pass input and output data to and from the compiling process. Process
and Pipe take care of serializing python objects and passing them across
process boundaries with pipes.
"""
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources as importlib_resources
from multiprocessing import connection
from os import environ as os_environ
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from . import exceptions
from .py_qssc import _compile_bytes, _compile_file, Diagnostic, ErrorCategory

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

    NONE = "none"  # Do not generate output, useful for testing
    QEM = "qem"
    QEQEM = "qe-qem"
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
    output_file: Union[Path, str, None] = None
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
class _CompilerStatus:
    """Internal compiler result status dataclass."""

    success: bool


def stringify_path(p):
    return str(p) if isinstance(p, Path) else p


class _CompilationManager:
    """Manager class to call compiler bindings from unique python process.

    TODO: It should not be necessary to call from a subprocess. This likely
    requires removing CLI argument parsing from pybind invocations of the compiler.
    """

    def __init__(self, compile_options: CompileOptions, return_diagnostics: bool):
        self.compile_options = compile_options
        self.return_diagnostics = return_diagnostics

    def _compile_call(
        self, args: List[str], on_diagnostic: Callable[[Diagnostic], Any]
    ) -> Tuple[bool, bytes]:
        """Implement for specific compilation pybind call."""
        raise NotImplementedError("A subclass must provide an implementation")

    def _compile_child_backend(
        self,
        on_diagnostic: Callable[[Diagnostic], Any],
    ) -> Tuple[_CompilerStatus, Union[bytes, None]]:
        # TODO: want a corresponding C++ interface to avoid overhead

        options = self.compile_options
        args = options.prepare_compiler_option_args()
        output_as_return = False if options.output_file else True

        # The qss-compiler expects the path to static resources in the environment
        # variable QSSC_RESOURCES. In the python package, those resources are
        # bundled under the directory resources/. Since python's functions for
        # looking up resources only treat files as resources, use the generated
        # python source _version.py to look up the path to the python package.
        with importlib_resources.path("qss_compiler", "_version.py") as version_py_path:
            resources_path = version_py_path.parent / "resources"
            os_environ["QSSC_RESOURCES"] = str(resources_path)
            success, output = self._compile_call(args, on_diagnostic)

        status = _CompilerStatus(success)
        if output_as_return and options.output_type is not OutputType.NONE:
            return status, output
        else:
            return status, None

    def _compile_child_runner(self, conn: connection.Connection) -> None:
        conn.recv()

        def on_diagnostic(diag):
            conn.send(diag)

        status, output = self._compile_child_backend(on_diagnostic)
        conn.send(status)
        if output is not None:
            conn.send_bytes(output)

    def compile(self) -> Union[bytes, str, None]:
        parent_side, child_side = mp_ctx.Pipe(duplex=True)

        try:
            childproc = mp_ctx.Process(target=self._compile_child_runner, args=(child_side,))
            childproc.start()

            parent_side.send(None)

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
                        if self.compile_options.on_diagnostic:
                            self.compile_options.on_diagnostic(received)
                        else:
                            diagnostics.append(received)
                    elif isinstance(received, _CompilerStatus):
                        success = received.success
                        break
                    else:
                        childproc.kill()
                        childproc.join()
                        raise exceptions.QSSCompilerCommunicationFailure(
                            "The compile process delivered an unexpected object instead of status "
                            "or diagnostic information. "
                            "This points to inconsistencies in the Python "
                            "interface code between the calling process and the compile process.",
                            return_diagnostics=self.return_diagnostics,
                        )

                if (
                    self.compile_options.output_file is None
                    and self.compile_options.output_type is not OutputType.NONE
                ):
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
                    return_diagnostics=self.return_diagnostics,
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
                    return_diagnostics=self.return_diagnostics,
                )

            # Place all higher-level diagnostics related to user input here
            # TODO: Best way to deal with multiple diagnostics?
            for diag in diagnostics:
                if diag.category == ErrorCategory.QSSCompilerSequenceTooLong:
                    raise exceptions.QSSCompilerSequenceTooLong(
                        diag.message,
                        diagnostics,
                        return_diagnostics=self.return_diagnostics,
                    )
                if diag.category == ErrorCategory.QSSControlSystemResourcesExceeded:
                    raise exceptions.QSSControlSystemResourcesExceeded(
                        diag.message,
                        diagnostics,
                        return_diagnostics=self.return_diagnostics,
                    )

            if not success:
                raise exceptions.QSSCompilationFailure(
                    "Failure during compilation",
                    diagnostics,
                    return_diagnostics=self.return_diagnostics,
                )

        except mp.ProcessError as e:
            raise exceptions.QSSCompilerError(
                "It's likely that you've hit a bug in the QSS Compiler. Please "
                "submit an issue to the team with relevant information "
                "(https://github.com/Qiskit/qss-compiler/issues):\n"
                f"{e}",
                return_diagnostics=self.return_diagnostics,
            )

        if self.compile_options.output_file is None:
            # return compilation result
            if self.compile_options.output_type == OutputType.MLIR:
                return output.decode("utf8")
            return output


class _CompileFile(_CompilationManager):
    def __init__(self, compile_options: CompileOptions, return_diagnostics: bool, input_file: str):
        super().__init__(compile_options, return_diagnostics)
        self.input_file = stringify_path(input_file)

    def _compile_call(
        self, args: List[str], on_diagnostic: Callable[[Diagnostic], Any]
    ) -> Tuple[bool, bytes]:
        return _compile_file(
            self.input_file,
            stringify_path(self.compile_options.output_file),
            args,
            on_diagnostic,
        )


class _CompileBytes(_CompilationManager):
    def __init__(
        self,
        compile_options: CompileOptions,
        return_diagnostics: bool,
        input: Union[str, bytes],
    ):
        super().__init__(compile_options, return_diagnostics)
        self.input = input

    def _compile_call(
        self, args: List[str], on_diagnostic: Callable[[Diagnostic], Any]
    ) -> Tuple[bool, bytes]:
        return _compile_bytes(
            self.input,
            stringify_path(self.compile_options.output_file),
            args,
            on_diagnostic,
        )


def _prepare_compile_options(
    compile_options: Optional[CompileOptions] = None, **kwargs
) -> CompileOptions:
    if compile_options is None:
        compile_options = CompileOptions(**kwargs)
    return compile_options


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
    compile_options = _prepare_compile_options(compile_options, **kwargs)
    return _CompileFile(compile_options, return_diagnostics, input_file).compile()


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
    compilation_manager = _CompileFile(compile_options, return_diagnostics, input_file)
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, compilation_manager.compile)


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
    return _CompileBytes(compile_options, return_diagnostics, input).compile()


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
    compilation_manager = _CompileBytes(compile_options, return_diagnostics, input)
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, compilation_manager.compile)

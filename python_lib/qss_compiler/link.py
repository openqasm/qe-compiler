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
This file defines the linker interface for the qss_compiler
package.

"""

from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from os import environ as os_environ
from typing import Mapping, Any, Optional, Callable, Union
import warnings

from .py_qssc import _link_file, Diagnostic, ErrorCategory
from .compile import stringify_path

from . import exceptions


@dataclass
class LinkOptions:
    """Options to the linker tool."""

    input_file: str = None
    """Path to input module."""
    input_bytes: Union[str, None] = None
    """Input payload as raw bytes"""
    output_file: Union[str, None] = None
    """Output file, if not supplied raw bytes will be returned."""
    target: str = None
    """Hardware target to select."""
    arguments: Mapping[str, Any] = field(default_factory=lambda: {})
    """Set the specific execution arguments of a pre-compiled program
        as a mapping of name to value.
    """
    config_path: str = ""
    """Target configuration path."""
    treat_warnings_as_errors: bool = True
    """Treat link warnings as errors"""
    on_diagnostic: Optional[Callable[[Diagnostic], Any]] = None
    """Optional callback for processing diagnostic messages from the linker."""
    number_of_threads: int = -1
    """Number of threads to use in linking
          -1 = number of cpus reported,
          0 disabled,
          > 1 = limit,
          defaults = -1
    """


def _prepare_link_options(link_options: Optional[LinkOptions] = None, **kwargs) -> LinkOptions:
    if link_options is None:
        link_options = LinkOptions(**kwargs)
    return link_options


def link_file(
    link_options: Optional[LinkOptions] = None,
    **kwargs,
) -> Optional[bytes]:
    """Link a module and bind arguments to create a payload.

    Consume a circuit module in a file and binds the provided circuit
    arguments, delivering a payload as output in a file.

    Args:
        input_file: Path to the circuit module to link.
        output_file: Path to write the output payload to.
        target: Compiler target to invoke for binding arguments (must match
            with the target that created the module).
        arguments: Circuit arguments as name/value map.

    Returns: Produces a payload in a file.
    """
    link_options = _prepare_link_options(link_options, **kwargs)

    input_file = stringify_path(link_options.input_file)
    output_file = stringify_path(link_options.output_file)
    config_path = stringify_path(link_options.config_path)

    diagnostics = []

    def on_diagnostic(diag):
        diagnostics.append(diag)

    if link_options.on_diagnostic is None:
        link_options.on_diagnostic = on_diagnostic

    for key, value in link_options.arguments.items():
        if not isinstance(value, float):
            if isinstance(value, int):
                link_options.arguments[key] = float(value)
            else:
                raise exceptions.QSSArgumentInputTypeError(
                    f"Only int & double arguments are supported, not {type(value)}"
                )

    if link_options.input_file is not None and link_options.input_bytes is not None:
        raise ValueError("only one of input_file or input_bytes should have a value")

    enable_in_memory = link_options.input_bytes is not None
    if enable_in_memory:
        input_file = link_options.input_bytes

    if output_file is None:
        output_file = ""

    # keep in mind that most of the infrastructure in the compile paths is for
    # taking care of the execution in a separate process. For the linker tool,
    # we aim at avoiding that right from the start!

    # The qss-compiler expects the path to static resources in the environment
    # variable QSSC_RESOURCES. In the python package, those resources are
    # bundled under the directory resources/. Since python's functions for
    # looking up resources only treat files as resources, use the generated
    # python source _version.py to look up the path to the python package.
    with importlib_resources.path("qss_compiler", "_version.py") as version_py_path:
        resources_path = version_py_path.parent / "resources"
        os_environ["QSSC_RESOURCES"] = str(resources_path)
        success, output = _link_file(
            input_file,
            enable_in_memory,
            output_file,
            link_options.target,
            config_path,
            link_options.arguments,
            link_options.treat_warnings_as_errors,
            link_options.on_diagnostic,
            link_options.number_of_threads,
        )
        if not success:
            exception_mapping = {
                ErrorCategory.QSSLinkerNotImplemented: exceptions.QSSLinkerNotImplemented,
                ErrorCategory.QSSLinkSignatureWarning: exceptions.QSSLinkSignatureWarning,
                ErrorCategory.QSSLinkSignatureError: exceptions.QSSLinkSignatureError,
                ErrorCategory.QSSLinkAddressError: exceptions.QSSLinkAddressError,
                ErrorCategory.QSSLinkSignatureNotFound: exceptions.QSSLinkSignatureNotFound,
                ErrorCategory.QSSLinkArgumentNotFoundWarning: exceptions.QSSLinkArgumentNotFoundWarning,  # noqa
                ErrorCategory.QSSLinkInvalidPatchTypeError: exceptions.QSSLinkInvalidPatchTypeError,
            }

            if diagnostics == [] or not isinstance(diagnostics[0], Diagnostic):
                pass
            elif diagnostics[0].category in exception_mapping.keys():
                raise exception_mapping[diagnostics[0].category](
                    diagnostics[0].message, diagnostics
                )
            raise exceptions.QSSLinkingFailure("Unknown linking failure", diagnostics)
        else:
            warning_mapping = {
                ErrorCategory.QSSLinkSignatureWarning: exceptions.QSSLinkSignatureWarning,
                ErrorCategory.QSSLinkArgumentNotFoundWarning: exceptions.QSSLinkArgumentNotFoundWarning,  # noqa
            }
            if diagnostics == [] or not isinstance(diagnostics[0], Diagnostic):
                pass
            else:
                for diagnostic in diagnostics:
                    if diagnostic.category in warning_mapping.keys():
                        warnings.warn(diagnostic.message, warning_mapping[diagnostic.category])

        # return in-memory raw bytes if output file is not specified
        if link_options.output_file is None:
            return output

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

from dataclasses import dataclass
from importlib import resources as importlib_resources
from os import environ as os_environ
from typing import Mapping, Any, Optional

from .py_qssc import _link_file
from .compile import _stringify_path


class QSSLinkingFailure(Exception):
    """Raised on compilation failure."""


@dataclass
class LinkOptions:
    """Options to the linker tool."""

    input_file: str
    """Path to input module."""
    output_file: str
    """Path to output payload."""
    target: str
    """Hardware target to select."""
    arguments: Mapping[str, Any]
    """Circuit arguments as mapping of name to value."""
    config_path: str = ""
    """Target configuration path."""


def _prepare_link_options(
    link_options: Optional[LinkOptions] = None, **kwargs
) -> LinkOptions:
    if link_options is None:
        link_options = LinkOptions(**kwargs)
    return link_options


def link_file(
    link_options: Optional[LinkOptions] = None,
    **kwargs,
):
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

    input_file = _stringify_path(link_options.input_file)
    output_file = _stringify_path(link_options.output_file)
    config_path = _stringify_path(link_options.config_path)

    for _, value in link_options.arguments.items():
        if not isinstance(value, float):
            raise QSSLinkingFailure(
                f"Only double arguments are supported, not {type(value)}"
            )

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
        success, errorMsg = _link_file(
            input_file, output_file, link_options.target, config_path,
            link_options.arguments
        )
        if not success:
            raise QSSLinkingFailure(errorMsg)

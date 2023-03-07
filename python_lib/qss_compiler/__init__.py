# (C) Copyright IBM 2022, 2023.
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
from ._version import version as __version__  # noqa: F401
from .py_qssc import __doc__  # noqa: F401

from .compile import (  # noqa: F401
    compile_file,
    compile_file_async,
    compile_str,
    compile_str_async,
    InputType,
    OutputType,
    CompileOptions,
    QSSCompilationFailure,
    QSSCompilerError,
)

from .py_qssc import (  # noqa: F401
    Diagnostic,
    ErrorCategory,
    Severity,
)

from .link import (  # noqa: F401
    link_file,
    LinkOptions,
    QSSLinkingFailure,
)

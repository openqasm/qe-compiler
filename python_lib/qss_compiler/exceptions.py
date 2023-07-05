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
This file contains all the exception subclasses.
"""

from typing import List, Optional

from .py_qssc import Diagnostic


def _diagnostics_to_str(diagnostics):
    return "\n".join([str(diag) for diag in diagnostics])


class QSSCompilerError(Exception):
    """Raised on errors invoking the compiler or when the interaction between
    Python interface and native backend code fails."""

    def __init__(self, *message: str, diagnostics: Optional[List[Diagnostic]] = None):
        """Set the error message."""
        self.message = message
        self.diagnostics = [] if diagnostics is None else diagnostics

    def __str__(self):
        """Return the message."""
        return "\n".join([self.message, _diagnostics_to_str(self.diagnostics)])


class QSSCompilerNoInputError(QSSCompilerError):
    """Raised when no input file or string is provided"""


class QSSCompilerCommunicationFailure(QSSCompilerError):
    """Raised on compilation communication failure."""


class QSSCompilerEOFFailure(QSSCompilerError):
    """Raised in case of EOF error."""


class QSSCompilerNonZeroStatus(QSSCompilerError):
    """Raised when non-zero status is returned."""


class QSSCompilationFailure(QSSCompilerError):
    """Raised during other compilation failure."""

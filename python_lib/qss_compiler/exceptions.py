# (C) Copyright IBM 2023, 2024.
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

from typing import Iterable, List, Optional

from .py_qssc import Diagnostic, ErrorCategory


def _diagnostics_to_str(diagnostics):
    return "\n".join([str(diag) for diag in diagnostics])


class QSSCompilerError(Exception):
    """Raised on errors invoking the compiler or when the interaction between
    Python interface and native backend code fails."""

    def __init__(
        self,
        message: str,
        diagnostics: Optional[List[Diagnostic]] = None,
        return_diagnostics: bool = False,
    ):
        """Set the error message."""
        self.message = message
        self.diagnostics = [] if diagnostics is None else diagnostics
        self.return_diagnostics = return_diagnostics

    def __str__(self):
        """Emit the exception as a string possibly including diagnostic messages."""
        if self.return_diagnostics:
            message_strings = [_diagnostics_to_str(self.diagnostics)]
            if self.message:
                message_strings.insert(0, self.message)
            return "\n".join(message_strings)
        return self.message


class QSSCompilerNoInputError(QSSCompilerError):
    """Raised when no input file or string is provided"""


class QSSCompilerCommunicationFailure(QSSCompilerError):
    """Raised on compilation communication failure."""


class QSSCompilerEOFFailure(QSSCompilerError):
    """Raised in case of EOF error."""


class QSSCompilerNonZeroStatus(QSSCompilerError):
    """Raised when non-zero status is returned."""


class QSSCompilerSequenceTooLong(QSSCompilerError):
    """Raised when input sequence is too long."""


class QSSCompilationFailure(QSSCompilerError):
    """Raised during other compilation failure."""


class QSSLinkingFailure(QSSCompilerError):
    """Raised on linking failure."""


class QSSLinkerNotImplemented(QSSCompilerError):
    """Raised on linking failure."""


class QSSArgumentInputTypeError(QSSLinkingFailure):
    """Raised when argument type is invalid"""


class QSSLinkSignatureError(QSSLinkingFailure):
    """Raised when signature file format is invalid"""


class QSSLinkSignatureWarning(QSSLinkingFailure, Warning):
    """Raised when signature file format is invalid but may still be processed"""


class QSSLinkAddressError(QSSLinkingFailure):
    """Raised when signature link address is invalid"""


class QSSLinkSignatureNotFound(QSSLinkingFailure):
    """Raised when argument signature file is not found"""


class QSSLinkArgumentNotFoundWarning(QSSLinkingFailure, Warning):
    """Raised when parameter name in signature is not found in arguments"""


class QSSLinkInvalidPatchTypeError(QSSLinkingFailure):
    """Raised when parameter patch type is invalid"""


class QSSLinkInvalidArgumentError(QSSLinkingFailure):
    """Raised when argument is invalid"""


class QSSControlSystemResourcesExceeded(QSSCompilerError):
    """Raised when control system resources (such as instruction memory) are exceeded."""


class OpenQASM3ParseFailure(QSSCompilerError):
    """Raised when a parser failure is received"""


class OpenQASM3UnsupportedInput(QSSCompilerError):
    """Raised when the input Openqasm 3 source uses semantics that are not supported by the
    compiler."""


def convert_diagnostics_to_exception(
    diagnostics: Iterable[Diagnostic], return_diagnostics: bool = False
) -> QSSCompilerError:
    """Convert diagnostics to Python compiler exceptions.

    Args:
        diagnostics: Iterable of diagnostics to check to raise exception from.
        return_diagnostics: Should the diagnostics be returned in the exception.
    Returns:
        Created exception (if any).

    """

    for diag in diagnostics:
        # Do not double print diagnostic message
        msg = "" if return_diagnostics else diag.message

        if diag.category == QSSCompilerSequenceTooLong:
            return QSSCompilerSequenceTooLong(
                msg,
                diagnostics,
                return_diagnostics=return_diagnostics,
            )
        if diag.category == ErrorCategory.QSSControlSystemResourcesExceeded:
            return QSSControlSystemResourcesExceeded(
                msg,
                diagnostics,
                return_diagnostics=return_diagnostics,
            )
        if diag.category == ErrorCategory.OpenQASM3ParseFailure:
            return OpenQASM3ParseFailure(
                msg,
                diagnostics,
                return_diagnostics=return_diagnostics,
            )
    return None


def raise_diagnostics(diagnostics: Iterable[Diagnostic], return_diagnostics: bool = False) -> None:
    """Convert diagnostics to Python exception if necessary and raise.

    Args:
        diagnostics: Iterable of diagnostics to check to raise exception from.
        return_diagnostics: Should the diagnostics be returned in the exception.
    Raises:
        QSSCompilerError: Raises the exception created from the diagnostic.

    """
    exception = convert_diagnostics_to_exception(diagnostics, return_diagnostics=return_diagnostics)
    if exception is not None:
        raise exception

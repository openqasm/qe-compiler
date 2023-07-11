//===- error.cpp  - Error reporting API -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the API for error reporting.
///
//===----------------------------------------------------------------------===//

#include "API/error.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace qssc {
static std::string_view getErrorCategoryAsString(ErrorCategory category) {
  using namespace qssc;
  switch (category) {
  case ErrorCategory::OpenQASM3ParseFailure:
    return "OpenQASM 3 parse error";

  case ErrorCategory::QSSCompilerError:
    return "Unknown compiler error";

  case ErrorCategory::QSSCompilerNoInputError:
    return "Error when no input file or string is provided";

  case ErrorCategory::QSSCompilerCommunicationFailure:
    return "Error on compilation communication failure";

  case ErrorCategory::QSSCompilerEOFFailure:
    return "EOF Error";

  case ErrorCategory::QSSCompilerNonZeroStatus:
    return "Errored because non-zero status is returned";

  case ErrorCategory::QSSCompilationFailure:
    return "Failure during compilation";

  case ErrorCategory::QSSLinkerNotImplemented:
    return "BindArguments not implemented for target";

  case ErrorCategory::QSSLinkSignatureError:
    return "Signature file format is invalid";

  case ErrorCategory::QSSLinkAddressError:
    return "Signature address is invalid";

  case ErrorCategory::QSSLinkSignatureNotFound:
    return "Signature file not found";

  case ErrorCategory::QSSLinkArgumentNotFoundWarning:
    return "Parameter in signature not found in arguments";

  case ErrorCategory::QSSLinkInvalidPatchTypeError:
    return "Invalid patch point type";

  case ErrorCategory::UncategorizedError:
    return "Compilation failure";
  }

  llvm_unreachable("unhandled category");
}

static llvm::StringRef getSeverityAsString(Severity sev) {
  switch (sev) {
  case Severity::Info:
    return "Info";
  case Severity::Warning:
    return "Warning";
  case Severity::Error:
    return "Error";
  case Severity::Fatal:
    return "Fatal";
  }

  llvm_unreachable("unhandled severity");
}

std::string Diagnostic::toString() const {
  std::string str;
  llvm::raw_string_ostream ostream(str);

  ostream << getSeverityAsString(severity) << ": "
          << getErrorCategoryAsString(category) << "\n";
  ostream << message;

  return str;
}

} // namespace qssc

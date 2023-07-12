//===- Error.h - Error Reporting API ----------------------------*- C++ -*-===//
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
///  This file declares the API for reporting errors.
///
//===----------------------------------------------------------------------===//

#ifndef QSS_COMPILER_ERROR_H
#define QSS_COMPILER_ERROR_H

#include "llvm/Support/Error.h"

#include <functional>
#include <optional>
#include <string>

namespace qssc {

enum class ErrorCategory {
  OpenQASM3ParseFailure,
  QSSCompilerError,
  QSSCompilerNoInputError,
  QSSCompilerCommunicationFailure,
  QSSCompilerEOFFailure,
  QSSCompilerNonZeroStatus,
  QSSCompilationFailure,
  QSSLinkerNotImplemented,
  QSSLinkSignatureError,
  QSSLinkAddressError,
  QSSLinkSignatureNotFound,
  QSSLinkArgumentNotFoundWarning,
  QSSLinkInvalidPatchTypeError,
  UncategorizedError,
};

enum class Severity {
  Info,
  Warning,
  Error,
  Fatal,
};

struct Diagnostic {
public:
  Severity severity;
  ErrorCategory category;
  std::string message; /// a detailed and actionable error message

  Diagnostic(Severity severity_, ErrorCategory category_, std::string message_)
      : severity(severity_), category(category_), message(std::move(message_)) {
  }

  std::string toString() const;
};

using DiagnosticCallback = std::function<void(const Diagnostic &)>;

llvm::Error emitDiagnostic(std::optional<DiagnosticCallback> onDiagnostic,
                           Severity severity, ErrorCategory category,
                           std::string message,
                           std::error_code ec = llvm::inconvertibleErrorCode());

} // namespace qssc

#endif // QSS_COMPILER_ERROR_H

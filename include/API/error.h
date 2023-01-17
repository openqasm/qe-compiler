//===- Error.h - Error Reporting API ----------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

#include <functional>
#include <string>

namespace qssc {

enum class ErrorCategory {
  OpenQASM3ParseFailure,
  UncategorizedError,
};

enum class Severity {
  Info,
  Warning,
  Error,
  Fatal,
};

struct Diagnostic {
  static std::string getErrorForCategory(qssc::ErrorCategory category);

public:
  Severity severity;
  ErrorCategory category;
  std::string error;   /// a brief description of the error
  std::string message; /// a detailed and actionable error message

  Diagnostic(Severity severity_, ErrorCategory category_,
             std::string &&message_)
      : severity(severity_), category(category_), message(std::move(message_)) {
    error = getErrorForCategory(category_);
  }
};

using DiagnosticCallback = std::function<void(const Diagnostic &)>;

} // namespace qssc

#endif // QSS_COMPILER_ERROR_H

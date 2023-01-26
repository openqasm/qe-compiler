//===- error.cpp  - Error reporting API -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

#include <string>

namespace qssc {
static std::string_view getErrorForCategory(ErrorCategory category) {
  using namespace qssc;
  switch (category) {
  case ErrorCategory::OpenQASM3ParseFailure:
    return "OpenQASM 3 parse error";

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
          << getErrorForCategory(category) << "\n";
  ostream << message;

  return str;
}

} // namespace qssc

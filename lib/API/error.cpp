//===- error.cpp  - Error reporting API -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
static std::string_view getErrorCategoryAsString(ErrorCategory category) {
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
          << getErrorCategoryAsString(category) << "\n";
  ostream << message;

  return str;
}

} // namespace qssc

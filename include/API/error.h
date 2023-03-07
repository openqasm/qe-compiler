//===- Error.h - Error Reporting API ----------------------------*- C++ -*-===//
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
///  This file declares the API for reporting errors.
///
//===----------------------------------------------------------------------===//

#ifndef QSS_COMPILER_ERROR_H
#define QSS_COMPILER_ERROR_H

#include <functional>
#include <string>
#include <string_view>

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

} // namespace qssc

#endif // QSS_COMPILER_ERROR_H

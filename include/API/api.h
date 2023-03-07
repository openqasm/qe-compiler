//===- api.h ----------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
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

#ifndef QSS_COMPILER_LIB_H
#define QSS_COMPILER_LIB_H

#include "API/error.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <unordered_map>

namespace qssc {

/// @brief Call the qss-compiler
/// @param argc the number of argument strings
/// @param argv array of argument strings
/// @param outputString an optional buffer for the compilation result
/// @param diagnosticCb an optional callback that will receive emitted
/// diagnostics
int compile(int argc, char const **argv, std::string *outputString,
            llvm::Optional<DiagnosticCallback> diagnosticCb);

llvm::Error
bindParameters(llvm::StringRef target, llvm::StringRef moduleInputPath,
               llvm::StringRef payloadOutputPath,
               std::unordered_map<std::string, double> const &parameters);

} // namespace qssc
#endif // QSS_COMPILER_LIB_H

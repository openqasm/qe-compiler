//===- api.h ----------------------------------------------------*- C++ -*-===//
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

#ifndef QSS_COMPILER_LIB_H
#define QSS_COMPILER_LIB_H

#include "API/error.h"
#include "Parameters/Parameters.h"

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

llvm::Error bindParameters(llvm::StringRef target,
                           llvm::StringRef configPath,
                           llvm::StringRef moduleInputPath,
                           llvm::StringRef payloadOutputPath,
                           qssc::parameters::ParameterSource &parameters);

} // namespace qssc
#endif // QSS_COMPILER_LIB_H

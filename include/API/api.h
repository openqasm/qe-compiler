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

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qssc {

/// @brief Call the qss-compiler
/// @param argc the number of argument strings
/// @param argv array of argument strings
/// @param outputString an optional buffer for the compilation result
/// @param diagnosticCb an optional callback that will receive emitted
/// diagnostics
int compile(int argc, char const **argv, std::string *outputString,
            std::optional<DiagnosticCallback> diagnosticCb);

/// @brief Call the parameter binder
/// @param target name of the target to employ
/// @param moduleInputPath path of the module to use as input
/// @param payloadOutputPath path of the payload to generate as output
/// @param parameters bindings for the parameters in the module to apply
/// @param errorMessage optional output for any occurring error message
/// @return 0 on success
int bindParameters(std::string_view target, std::string_view moduleInputPath,
                   std::string_view payloadOutputPath,
                   std::unordered_map<std::string, double> const &parameters,
                   std::string *errorMessage);

} // namespace qssc
#endif // QSS_COMPILER_LIB_H

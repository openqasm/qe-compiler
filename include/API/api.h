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
/// @param arguments bindings for the parameters in the module to apply
/// @param treatWarningsAsErrors return errors in place of warnings
/// @param diagnosticCb an optional callback that will receive emitted
/// diagnostics
/// @return 0 on success
int bindArguments(std::string_view target, std::string_view configPath,
                  std::string_view moduleInputPath,
                  std::string_view payloadOutputPath,
                  std::unordered_map<std::string, double> const &arguments,
                  bool treatWarningsAsErrors,
                  const std::optional<DiagnosticCallback> &onDiagnostic);

} // namespace qssc
#endif // QSS_COMPILER_LIB_H

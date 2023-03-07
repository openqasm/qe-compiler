//===- OpenQASM3Frontend.h --------------------------------------*- C++ -*-===//
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
/// \file
/// This file declares the interface to the OpenQASM 3 frontend.
///
//===----------------------------------------------------------------------===//

#ifndef OPENQASM3_FRONTEND_H
#define OPENQASM3_FRONTEND_H

#include "API/error.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"

namespace qssc::frontend::openqasm3 {

/// @brief Parse an OpenQASM 3 source file and emit high-level IR in the
/// OpenQASM 3 dialect or dump the AST
/// @param source input source as string or filename of the source
/// @param sourceIsFilename true when the parameter source is the name of a
/// source file, false when the parameter is the source input
/// @param emitRawAST whether the raw AST should be dumped
/// @param emitPrettyAST whether a pretty-printed AST should be dumped
/// @param emitMLIR whether high-level IR should be emitted
/// @param newModule ModuleOp container for emitting MLIR into
/// @param diagnosticCb a callback that will receive emitted diagnostics
/// @return an llvm::Error in case of failure, or llvm::Error::success()
/// otherwise
llvm::Error parse(std::string const &source, bool sourceIsFilename,
                  bool emitRawAST, bool emitPrettyAST, bool emitMLIR,
                  mlir::ModuleOp &newModule,
                  llvm::Optional<DiagnosticCallback> diagnosticCb);

}; // namespace qssc::frontend::openqasm3

#endif // OPENQASM3_FRONTEND_H

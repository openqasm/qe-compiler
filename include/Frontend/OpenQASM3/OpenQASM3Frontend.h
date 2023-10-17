//===- OpenQASM3Frontend.h --------------------------------------*- C++ -*-===//
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
/// \file
/// This file declares the interface to the OpenQASM 3 frontend.
///
//===----------------------------------------------------------------------===//

#ifndef OPENQASM3_FRONTEND_H
#define OPENQASM3_FRONTEND_H

#include "API/errors.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <optional>

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
                  std::optional<DiagnosticCallback> diagnosticCb);

}; // namespace qssc::frontend::openqasm3

#endif // OPENQASM3_FRONTEND_H

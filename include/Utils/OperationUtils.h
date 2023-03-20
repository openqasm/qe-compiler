//===- OperationUtils.h -----------------------------------------*- C++ -*-===//
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
///  Utility functions handling MLIR Operations.
///
//===----------------------------------------------------------------------===//

#ifndef UTILS_OPERATIONUTILS_H
#define UTILS_OPERATIONUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace qssc::utils {

/// Create and return an inline assembly operation, as a convenience wrapper
/// that does not require all parameter's
///
/// @param builder an MLIR OpBuilder to create the operation with
/// @param loc the source location information for the new operation
/// @param resultTypes the types of results to deliver
/// @param operands the operands to pass into the inline assembly
/// @param assemblyString the assembly code
/// @param constraints the constraints for mapping operands and results to the
/// assembly
/// @param hasSideEffects whether the assembly code should be marked as having
/// side effects
inline mlir::LLVM::InlineAsmOp
createInlineAsmOp(mlir::OpBuilder builder, mlir::Location loc,
                  mlir::TypeRange resultTypes, mlir::ValueRange operands,
                  llvm::StringRef assemblyString,
                  const llvm::Twine &constraints, bool hasSideEffects = false) {
  return builder.create<mlir::LLVM::InlineAsmOp>(
      loc, resultTypes, operands, builder.getStringAttr(assemblyString),
      builder.getStringAttr(constraints), hasSideEffects,
      /*is_align_stack=*/false, mlir::LLVM::AsmDialectAttr(nullptr),
      /*operand_attrs=*/mlir::ArrayAttr());
}
} // namespace qssc::utils

#endif

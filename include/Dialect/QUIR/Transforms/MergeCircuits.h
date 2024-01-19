//===- MergeCircuits.h - Merge circuits ops ---------------------*- C++ -*-===//
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
///  This file declares the pass for merging back-to-back circuits into a single
///  circuit when possible.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_MERGE_CIRCUITS_H
#define QUIR_MERGE_CIRCUITS_H

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// @brief Merge together back to back circuits into a single circuit
struct MergeCircuitsPass
    : public PassWrapper<MergeCircuitsPass, OperationPass<>> {
  void runOnOperation() override;

  static CircuitOp getCircuitOp(CallCircuitOp callCircuitOp,
                                llvm::StringMap<Operation *> *symbolMap);
  static LogicalResult mergeCallCircuits(
      MLIRContext *context, PatternRewriter &rewriter,
      CallCircuitOp callCircuitOp, CallCircuitOp nextCallCircuitOp,
      llvm::StringMap<Operation *> *symbolMap,
      std::optional<llvm::SmallVector<Operation *>> barriers = std::nullopt);

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
}; // struct MergeCircuitsPass
} // namespace mlir::quir
#endif // QUIR_MERGE_CIRCUITS_H

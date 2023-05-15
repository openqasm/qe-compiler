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
///  This file declares the pass for merging back-to-back circuits into single
///  circuit when possible.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_MERGE_CIRCUITS_H
#define QUIR_MERGE_CIRCUITS_H

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Pass/Pass.h"
#include <mlir/IR/PatternMatch.h>

namespace mlir::quir {

/// @brief Merge together measures in a circuit that are lexicographically
/// adjacent into a single variadic measurement.
struct MergeCircuitsPass
    : public PassWrapper<MergeCircuitsPass, OperationPass<>> {
  void runOnOperation() override;

  static CircuitOp getCircuitOp(CallCircuitOp callCircuitOp);
  static LogicalResult mergeCallCircuits(PatternRewriter &rewriter,
                                CallCircuitOp callCircuitOp,
                                CallCircuitOp nextCallCircuitOp);

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MergeCircuitsPass
} // namespace mlir::quir
#endif // QUIR_MERGE_CIRCUITS_H

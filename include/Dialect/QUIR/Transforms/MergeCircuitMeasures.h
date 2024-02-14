//===- MergeCircuitMeasures.h - Merge measurement ops in circuits - C++ -*-===//
//
// (C) Copyright IBM 2024.
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
///  This file declares the pass for merging measurements inside circuits into a
///  single measure op
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_MERGE_CIRCUIT_MEASURES_H
#define QUIR_MERGE_CIRCUIT_MEASURES_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// @brief Merge together measures in a circuit that are topologically
/// adjacent into a single variadic measurement.
struct MergeCircuitMeasuresTopologicalPass
    : public PassWrapper<MergeCircuitMeasuresTopologicalPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
}; // struct MergeCircuitMeasuresTopologicalPass

} // namespace mlir::quir

#endif // QUIR_MERGE_CIRCUIT_MEASURES_H

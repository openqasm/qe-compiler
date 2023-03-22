//===- ReorderMeasurements.h - Move measurement ops later -------*- C++ -*-===//
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
///  This file declares the pass for moving measurements as late as possible
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_REORDER_MEASURES_H
#define QUIR_REORDER_MEASURES_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// @brief Move measures in a circuit to be as late as possible topologically
struct ReorderMeasurementsPass
    : public PassWrapper<ReorderMeasurementsPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct ReorderMeasurementsPass

} // namespace mlir::quir

#endif // QUIR_REORDER_MEASURES_H

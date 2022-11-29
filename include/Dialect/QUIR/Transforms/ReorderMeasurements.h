//===- ReorderMeasurements.h - Move measurement ops later -------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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

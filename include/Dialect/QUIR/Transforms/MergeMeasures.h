//===- MergeMeasures.h - Merge measurement ops ------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for merging measurements into a single measure
///  op
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_MERGE_MEASURES_H
#define QUIR_MERGE_MEASURES_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {
struct MergeMeasuresPass
    : public PassWrapper<MergeMeasuresPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MergeMeasuresPass
} // namespace mlir::quir

#endif // QUIR_MERGE_MEASURES_H

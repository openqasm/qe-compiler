//===- SlicePorts.h - Slice mlir using ports ----------------------*-
// C++-*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for slicing mlir by port as a strategy.
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_SLICE_PORT_H
#define PULSE_SLICE_PORT_H

#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

class SlicePortPass
    : public PassWrapper<SlicePortPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};
} // namespace mlir::pulse

#endif // PULSE_SLICE_PORT_H

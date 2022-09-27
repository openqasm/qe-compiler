//===- PortGroupPrune.h - Remove extra pulse ops. *--C++-*-----------------===//
//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for removing port groups and select port ops.
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_PORTGROUP_PRUNE_H
#define PULSE_PORTGROUP_PRUNE_H

#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

class PortGroupPrunePass
    : public PassWrapper<PortGroupPrunePass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};
} // namespace mlir::pulse

#endif // PULSE_PORTGROUP_PRUNE_H

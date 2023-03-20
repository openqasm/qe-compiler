//===- PortGroupPrune.cpp - Remove extra pulse ops. *--C++-*---------------===//
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
///  This file implements the pass for removing port groups and select port ops.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/PortGroupPrune.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir::pulse {

void PortGroupPrunePass::runOnOperation() {

  auto module = getOperation();

  for (auto function : llvm::make_early_inc_range(module.getOps<FuncOp>())) {

    for (auto selectOp :
         llvm::make_early_inc_range(function.getOps<pulse::Port_SelectOp>())) {

      auto id = selectOp.id();
      for (auto portCreateOp : function.getOps<pulse::Port_CreateOp>()) {
        if (portCreateOp.uid() == id) {
          for (auto &use : llvm::make_early_inc_range(selectOp->getUses()))
            use.set(portCreateOp.out());
          selectOp->erase();
          break;
        }
      }
    }
    for (auto portGroupOp : llvm::make_early_inc_range(
             function.getOps<pulse::PortGroup_CreateOp>()))
      portGroupOp.erase();
  }
}

llvm::StringRef PortGroupPrunePass::getArgument() const {
  return "pulse-prune";
}
llvm::StringRef PortGroupPrunePass::getDescription() const {
  return "Prune port groups.";
}
} // namespace mlir::pulse

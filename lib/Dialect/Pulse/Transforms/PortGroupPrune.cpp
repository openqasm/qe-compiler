//===- PortGroupPrune.cpp - Remove extra pulse ops. *--C++-*---------------===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

//===- SlicePorts.cpp - Slice mlir using ports -------------------*- C++-*-===//
//
// (C) Copyright IBM 2022, 2023.
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
///  This file implements the pass for slicing mlir by port as a strategy.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/SlicePorts.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"

namespace mlir::pulse {

void gatherDependencies(Operation *key, Operation *op,
                        std::vector<Operation *> &dependencies) {
  if (key != op) {
    if (op->getNumOperands() == 0) {
      dependencies.push_back(op);
    } else {
      for (auto operand : op->getOperands())
        gatherDependencies(key, operand.getDefiningOp(), dependencies);
      dependencies.push_back(op);
    }
  }
}

void SlicePortPass::runOnOperation() {

  auto module = getOperation();

  auto functions = module.getOps<FuncOp>();

  if (std::distance(functions.begin(), functions.end()) != 1) {
    signalPassFailure();
    llvm::errs() << "Unable to process MLIRs with more than one FuncOp"
                 << "\n";
    return;
  }

  auto function = *functions.begin();

  for (auto portOp :
       llvm::make_early_inc_range(function.getOps<pulse::Port_CreateOp>())) {

    if (portOp->getUsers().empty()) {
      portOp->erase();
      continue;
    }

    OpBuilder builder(function);

    auto functionOp = builder.create<FuncOp>(
        function->getLoc(), portOp.uid(),
        FunctionType::get(function->getContext(), TypeRange{}, TypeRange{}));

    builder = builder.atBlockBegin(functionOp.addEntryBlock());

    BlockAndValueMapping mapper;
    builder.clone(*portOp, mapper);

    for (auto *user : llvm::make_early_inc_range(portOp->getUsers())) {

      std::vector<Operation *> dependencies;
      gatherDependencies(portOp, user, dependencies);
      for (auto *op : dependencies)
        builder.clone(*op, mapper);

      for (auto rit = dependencies.rbegin(); rit != dependencies.rend();
           ++rit) {
        dependencies.erase(std::next(rit).base());
        (*rit)->erase();
      }
    }
    builder.create<mlir::ReturnOp>(functionOp.getLoc());
    portOp->erase();
  }
}

llvm::StringRef SlicePortPass::getArgument() const { return "pulse-slice"; }
llvm::StringRef SlicePortPass::getDescription() const {
  return "Slice mlir using ports.";
}
} // namespace mlir::pulse

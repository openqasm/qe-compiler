//===- RemoveUnusedCircuits.h - Remove Unused Circuits  ----------- C++ -*-===//
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
///  This file implements the pass for removing unused circuits
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/RemoveUnusedCircuits.h"

#include "Dialect/QUIR/IR/QUIROps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <mlir/IR/Operation.h>

using namespace mlir;
using namespace mlir::quir;

void RemoveUnusedCircuitsPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  llvm::StringSet<> calledCircuits;

  moduleOperation->walk([&](CallCircuitOp callCircuitOp) {
    calledCircuits.insert(callCircuitOp.getCallee());
  });

  llvm::SmallVector<Operation *> eraseList;

  moduleOperation->walk([&](CircuitOp circuitOp) {
    auto search = calledCircuits.find(circuitOp.getSymName());
    if (search == calledCircuits.end())
      eraseList.push_back(circuitOp.getOperation());
  });

  for (auto *op : eraseList)
    op->erase();
} // runOnOperation

llvm::StringRef RemoveUnusedCircuitsPass::getArgument() const {
  return "remove-unused-circuits";
}
llvm::StringRef RemoveUnusedCircuitsPass::getDescription() const {
  return "Remove unused circuits from a module";
}

llvm::StringRef RemoveUnusedCircuitsPass::getName() const {
  return "Remove Unused Circuits Pass";
}

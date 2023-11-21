//===- RemoveQubitOperands.h - Remove qubit args -----------------*- C++-*-===//
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
//
//  This file declares the pass for removing qubit arguments from subroutines
//  and subroutine calls, replacing the arguments with qubit declarations inside
//  the subroutine body.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_REMOVE_QUBIT_ARGS_H
#define QUIR_REMOVE_QUBIT_ARGS_H

#include <deque>
#include <unordered_set>

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::quir {
struct RemoveQubitOperandsPass
    : public PassWrapper<RemoveQubitOperandsPass, OperationPass<>> {
  auto lookupQubitId(const Value val) -> int;
  void addQubitDeclarations(mlir::func::FuncOp funcOp);
  void processCallOp(Operation *op);
  void runOnOperation() override;

  std::deque<Operation *> callWorkList;
  std::unordered_set<Operation *> clonedFuncs;
  std::unordered_set<Operation *> alreadyProcessed;
  Operation *moduleOperation;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct SubroutineCloningPass
} // namespace mlir::quir

#endif // QUIR_REMOVE_QUBIT_ARGS_H

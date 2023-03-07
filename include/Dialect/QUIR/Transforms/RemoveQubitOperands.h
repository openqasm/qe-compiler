//===- RemoveQubitOperands.h - Remove qubit args -----------------*- C++-*-===//
//
// (C) Copyright IBM 2021 - 2023
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
  void addQubitDeclarations(FuncOp funcOp);
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

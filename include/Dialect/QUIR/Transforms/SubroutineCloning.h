//===- SubroutineCloning.cpp - Resolve subroutine calls ---------*- C++ -*-===//
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
//
//  This file declares the pass for cloning subroutines, resolving all qubit
//  arguments and mangling subroutine names with the id of the arguments from
//  each call. Thus each cloned subroutine def matches with a call that has a
//  particular set of qubit arguments. The call is updated to match the newly
//  cloned def.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_SUBROUTINE_CLONING_H
#define QUIR_SUBROUTINE_CLONING_H

#include <deque>
#include <unordered_set>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::quir {
struct SubroutineCloningPass
    : public PassWrapper<SubroutineCloningPass, OperationPass<>> {
  auto lookupQubitId(const Value val) -> int;
  auto getMangledName(Operation *op) -> std::string;
  void processCallOp(Operation *op);
  void runOnOperation() override;

  std::deque<Operation *> callWorkList;
  std::unordered_set<Operation *> clonedFuncs;
  Operation *moduleOperation;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct SubroutineCloningPass
} // namespace mlir::quir

#endif // QUIR_SUBROUTINE_CLONING_H

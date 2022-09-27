//===- SubroutineCloning.cpp - Resolve subroutine calls ---------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
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

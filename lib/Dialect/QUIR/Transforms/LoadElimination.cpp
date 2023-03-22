//===- LoadElimination.cpp - Remove unnecessary loads -----------*- C++ -*-===//
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
///  This file implements the pass for replacing unnecessary variable loads.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/LoadElimination.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::quir {

void LoadEliminationPass::runOnOperation() {
  // Eliminate simple cases where variables are assigned only once. That is,
  // they are effectively constants. (to be extended considerably as part of
  // IBM-Q-Software/qss-compiler#470)
  //
  // We consider only declarations for variables that have only a single
  // assignment. Then, we iterate all uses of the variable. When a use of a
  // variable always happens after the assignment, then we can forward the
  // assigned value to the variable's use and eliminate the quir::VariableUse
  // operation.
  //
  // This code is strongly inspired by the concepts behind
  // mlir::affineScalarReplace().
  Operation *op = getOperation();
  SymbolTableCollection symbolTable_;
  SymbolUserMap symbolUsers(symbolTable_, op);
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  SmallVector<Operation *, 4> varUsesToErase;

  op->walk([&](mlir::oq3::DeclareVariableOp decl) {
    // Each variable must only have a single assigment statement

    auto symbolUses = symbolUsers.getUsers(decl);

    Operation *assignment = nullptr;

    auto numAssignments = std::count_if(
        symbolUses.begin(), symbolUses.end(), [&](Operation *userOp) {
          if (mlir::isa<mlir::oq3::VariableAssignOp>(userOp) ||
              mlir::isa<mlir::oq3::CBitAssignBitOp>(userOp)) {
            // TODO have a common interface that identifies any
            // assignment to a variable
            assignment = userOp;
            return true;
          }
          return false;
        });

    if (numAssignments > 1)
      return WalkResult::advance();

    // only support assignment by VariableAssignOp, for now
    if (!mlir::isa<mlir::oq3::VariableAssignOp>(assignment))
      return WalkResult::advance();

    auto varAssignmentOp = mlir::cast<mlir::oq3::VariableAssignOp>(assignment);

    // Transfer marker for input parameters
    // Note: for arith.constant operations, canonicalization will drop these
    // attributes and we need to find another way (to be specific:
    // canonicalization will move constants to the begin of ops like Functions
    // by means of dialect->materializeConstant(...) that creates new
    // constants). For now and for angle constants, this approach is good-enough
    // while not satisfying.
    if (decl.isInputVariable())
      varAssignmentOp.assigned_value().getDefiningOp()->setAttr(
          mlir::quir::getInputParameterAttrName(), decl.getNameAttr());

    for (auto *userOp : symbolUses) {

      if (!mlir::isa<mlir::oq3::VariableLoadOp>(userOp))
        continue;

      auto variableUse = mlir::cast<mlir::oq3::VariableLoadOp>(userOp);

      // If this use of the variable is always executed after the assignment,
      // then the variable will have the assigned value at the time of this
      // use and we can replace this use with the assigned value. Note that
      // there is only ever one assignment to the variables considered, so the
      // order of assignments does not matter.
      if (!domInfo.dominates(assignment, variableUse)) // that is not the case
        continue;

      variableUse.replaceAllUsesWith(varAssignmentOp.assigned_value());
      varUsesToErase.push_back(variableUse);
    }

    return WalkResult::advance();
  });

  for (auto *op : varUsesToErase)
    op->erase();
}

llvm::StringRef LoadEliminationPass::getArgument() const {
  return "quir-eliminate-loads";
}

llvm::StringRef LoadEliminationPass::getDescription() const {
  return "Eliminate variable loads by forwarding the operands of assignments "
         "to a variable to subsequent uses of the variable.";
}

} // namespace mlir::quir

//===- UnusedVariable.cpp - Remove unused variables -------------*- C++ -*-===//
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
///  This file implements the pass for removing variables that are unused
///  by any subsequent load
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/UnusedVariable.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"

#include <utility>

using namespace mlir;
using namespace quir;
using namespace oq3;

///
/// \brief Entry point for the pass.
void UnusedVariablePass::runOnOperation() {
  mlir::SymbolTableCollection symbolTable;
  mlir::SymbolUserMap symbolUsers(symbolTable, getOperation());

  getOperation()->walk([&](DeclareVariableOp declOp) {
    if (declOp.isOutputVariable())
      return mlir::WalkResult::advance();

    // iterate through uses
    for (auto *useOp : symbolUsers.getUsers(declOp)) {
      if (auto useVariable = dyn_cast<VariableLoadOp>(useOp)) {
        if (!useVariable || !useVariable.use_empty())
          return mlir::WalkResult::advance();
      }
    }

    // No uses found, so now we can erase all references (just stores) and the
    // declaration
    for (auto *useOp : symbolUsers.getUsers(declOp))
      useOp->erase();
    ;

    declOp->erase();

    return mlir::WalkResult::advance();
  });
}

llvm::StringRef UnusedVariablePass::getArgument() const {
  return "remove-unused-variables";
}
llvm::StringRef UnusedVariablePass::getDescription() const {
  return "Remove variables that are not outputs and do not have any loads/uses";
}

llvm::StringRef UnusedVariablePass::getName() const {
  return "Unused Variable Pass";
}

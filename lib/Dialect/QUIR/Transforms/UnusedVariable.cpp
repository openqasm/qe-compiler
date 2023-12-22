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
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace quir;
using namespace oq3;

namespace {
/// This pattern matches on variable declarations that are not marked 'output'
/// and are not followed by a use of the same variable, and removes them
struct UnusedVariablePat : public OpRewritePattern<DeclareVariableOp> {
  UnusedVariablePat(MLIRContext *context, mlir::SymbolUserMap &symbolUses)
      : OpRewritePattern<DeclareVariableOp>(context, /*benefit=*/1),
        symbolUses(symbolUses) {}
  mlir::SymbolUserMap &symbolUses;
  LogicalResult
  matchAndRewrite(DeclareVariableOp declOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (declOp.isOutputVariable())
      return failure();

    // iterate through uses
    for (auto *useOp : symbolUses.getUsers(declOp)) {
      if (auto useVariable = dyn_cast<VariableLoadOp>(useOp)) {
        if (!useVariable || !useVariable.use_empty())
          return failure();
      }
    }

    // No uses found, so now we can erase all references (just stores) and the
    // declaration
    for (auto *useOp : symbolUses.getUsers(declOp))
      rewriter.eraseOp(useOp);

    rewriter.eraseOp(declOp);
    return success();
  } // matchAndRewrite

}; // struct UnusedVariablePat
} // anonymous namespace

///
/// \brief Entry point for the pass.
void UnusedVariablePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  mlir::GreedyRewriteConfig config;
  mlir::SymbolTableCollection symbolTable;
  mlir::SymbolUserMap symbolUsers(symbolTable, getOperation());

  // use cheaper top-down traversal (in this case, bottom-up would not behave
  // any differently)
  config.useTopDownTraversal = true;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  patterns.add<UnusedVariablePat>(&getContext(), symbolUsers);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                          config)))
    signalPassFailure();
}

llvm::StringRef UnusedVariablePass::getArgument() const {
  return "remove-unused-variables";
}
llvm::StringRef UnusedVariablePass::getDescription() const {
  return "Remove variables that are not outputs and do not have any loads/uses";
}

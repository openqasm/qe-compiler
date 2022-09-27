//===- UnusedVariable.cpp - Remove unused variables -------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace quir;

namespace {
/// This pattern matches on variable declarations that are not marked 'output'
/// and are not followed by a use of the same variable, and removes them
struct UnusedVariablePat : public OpRewritePattern<DeclareVariableOp> {
  UnusedVariablePat(MLIRContext *context)
      : OpRewritePattern<DeclareVariableOp>(context, /*benefit=*/1) {}
  LogicalResult
  matchAndRewrite(DeclareVariableOp declOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (declOp.isOutputVariable())
      return failure();

    // iterate through uses
    llvm::Optional<SymbolTable::UseRange> variableSymbolUses =
        SymbolTable::getSymbolUses(declOp.sym_nameAttr(),
                                   declOp->getParentRegion());
    if (variableSymbolUses.hasValue()) { // some variable references exist
      for (auto use : variableSymbolUses.getValue()) {
        if (auto useVariable = dyn_cast<UseVariableOp>(use.getUser())) {
          if (!useVariable.use_empty())
            return failure();
        }
      }

      // No uses found, so now we can erase all references (just stores) and the
      // declaration
      for (auto use : variableSymbolUses.getValue())
        rewriter.eraseOp(use.getUser());
    }

    rewriter.eraseOp(declOp);
    return success();
  } // matchAndRewrite
};  // struct UnusedVariablePat
} // anonymous namespace

///
/// \brief Entry point for the pass.
void UnusedVariablePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  mlir::GreedyRewriteConfig config;

  // use cheaper top-down traversal (in this case, bottom-up would not behave
  // any differently)
  config.useTopDownTraversal = true;
  patterns.insert<UnusedVariablePat>(&getContext());

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

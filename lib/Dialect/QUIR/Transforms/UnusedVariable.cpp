//===- UnusedVariable.cpp - Remove unused variables -------------*- C++ -*-===//
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
      if (auto useVariable = dyn_cast<UseVariableOp>(useOp)) {
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
  patterns.insert<UnusedVariablePat>(&getContext(), symbolUsers);

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

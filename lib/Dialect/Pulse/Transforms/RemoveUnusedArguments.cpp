//===- RemoveUnusedArguments.cpp -- remove unused args from call *- C++ -*-===//
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
///  This file implements the pass for removing unused arguments from
///  pulse.call_sequence and pulse.sequence
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/RemoveUnusedArguments.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include <vector>

#define DEBUG_TYPE "RemoveUnusedArguments"

using namespace mlir;
using namespace mlir::pulse;

namespace {

struct RemoveUnusedArgumentsPattern
    : public OpRewritePattern<pulse::CallSequenceOp> {
  explicit RemoveUnusedArgumentsPattern(MLIRContext *ctx)
      : OpRewritePattern<pulse::CallSequenceOp>(ctx) {}
  LogicalResult matchAndRewrite(pulse::CallSequenceOp callSequenceOp,
                                PatternRewriter &rewriter) const override {

    LLVM_DEBUG(llvm::errs() << "Matching: ");
    LLVM_DEBUG(callSequenceOp.dump());

    Operation *findOp = SymbolTable::lookupNearestSymbolFrom<SequenceOp>(
        callSequenceOp, callSequenceOp.calleeAttr());

    if (!findOp)
      return failure();

    SequenceOp sequenceOp = dyn_cast<SequenceOp>(findOp);

    llvm::BitVector argIndicesBV(sequenceOp.getNumArguments());
    std::vector<Operation *> testEraseList;

    for (auto const &argumentResult :
         llvm::enumerate(sequenceOp.getArguments())) {
      if (argumentResult.value().use_empty()) {
        auto index = argumentResult.index();
        argIndicesBV.set(index);
        LLVM_DEBUG(llvm::errs() << "Removing argument: ");
        LLVM_DEBUG(argumentResult.value().dump());

        auto *argOp = callSequenceOp.getOperand(index).getDefiningOp();
        testEraseList.push_back(argOp);
      }
    }

    // indicate match failure if there are no unused arguments
    if (argIndicesBV.none())
      return failure();

    // rewrite - removed arguments and matching operands
    sequenceOp.eraseArguments(argIndicesBV);
    callSequenceOp->eraseOperands(argIndicesBV);

    // remove defining ops if the have no usage
    for (auto *argOp : testEraseList)
      if (argOp->use_empty())
        rewriter.eraseOp(argOp);

    return success();
  }
}; // struct RemoveUnusedArgumentsPattern

} // end anonymous namespace

void RemoveUnusedArgumentsPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.insert<RemoveUnusedArgumentsPattern>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

llvm::StringRef RemoveUnusedArgumentsPass::getArgument() const {
  return "pulse-remove-unused-arguments";
}

llvm::StringRef RemoveUnusedArgumentsPass::getDescription() const {
  return "Remove sequence arguments that are unused by physical channel";
}

//===- MergeDelays.cpp - merges quir delays on the same  --------*- C++ -*-===//
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
///  This file implements the pass for merging back to back quir.delays.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/Transforms/MergeDelays.h"
#include "Dialect/QCS/IR/QCSOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::qcs;

namespace {
// This pattern matches on two DelayOps that back to back and merges
// into a single DelayOp with the sum of the durations
// the ignoreTarget flag may be used to enforce target equality
// when merging the delays
struct DelayAndDelayPattern : public OpRewritePattern<DelayCyclesOp> {
  explicit DelayAndDelayPattern(MLIRContext *ctx)
      : OpRewritePattern<DelayCyclesOp>(ctx) {}

  LogicalResult matchAndRewrite(DelayCyclesOp delayOp,
                                PatternRewriter &rewriter) const override {

    // get next operation and test for Delay
    Operation *nextOp = delayOp->getNextNode();
    if (!nextOp)
      return failure();

    auto nextDelayOp = dyn_cast<DelayCyclesOp>(nextOp);
    if (!nextDelayOp)
      return failure();

    auto mergeDelayValue = delayOp.time() + nextDelayOp.time();

    // set first DelayOp duration to summed constant
    delayOp->setAttr("time", rewriter.getI64IntegerAttr(mergeDelayValue));

    // erase following delay
    rewriter.eraseOp(nextOp);

    return success();

  } // matchAndRewrite
};  // struct DelayAndDelayPattern
} // end anonymous namespace

void MergeQCSDelayPass::runOnOperation() {

  Operation *operation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<DelayAndDelayPattern>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(operation, std::move(patterns))))
    signalPassFailure();

} // runOnOperation

llvm::StringRef MergeQCSDelayPass::getArgument() const {
  return "qcs-merge-delay";
}

llvm::StringRef MergeQCSDelayPass::getDescription() const {
  return "Merge sequencial qcs.delays";
}

//===- MergeDelays.cpp - merges delays on the same target -------*- C++ -*-===//
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
///  This file implements the pass for merging back to back pulse.delays.
///  The current implementation defaults to ignoring the target, there
///  is an option (ignoreTarget) to override this.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/MergeDelays.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::pulse;

namespace {
// This pattern matches on two DelayOps that back to back and merges
// into a single DelayOp with the sum of the durations
// the ignoreTarget flag may be used to enforce target equality
// when merging the delays
struct DelayAndDelayPattern : public OpRewritePattern<DelayOp> {
  explicit DelayAndDelayPattern(MLIRContext *ctx)
      : OpRewritePattern<DelayOp>(ctx) {}

  LogicalResult matchAndRewrite(DelayOp delayOp,
                                PatternRewriter &rewriter) const override {

    // TODO: determine how to pass ignoreTarget to the pass as an option
    bool ignoreTarget = false;

    // get next operation and test for Delay
    Operation *nextOp = delayOp->getNextNode();
    if (!nextOp)
      return failure();

    auto nextDelayOp = dyn_cast<DelayOp>(nextOp);
    if (!nextDelayOp)
      return failure();

    // found a delay and a delay
    // verify port | frame (second operand) is the same
    // this verification will be ignored if ignoreTarget is set to true
    auto firstDelayPortOrFrame = delayOp.getTarget();
    auto secondDelayPortOrFrame = nextDelayOp.getTarget();

    if (!ignoreTarget && (firstDelayPortOrFrame != secondDelayPortOrFrame))
      return failure();

    // get delay times and sum as int 32
    // TODO: check to see if int 32 is sufficient
    auto firstDelay = delayOp.getDur();
    auto secondDelay = nextDelayOp.getDur();

    auto firstDelayOp =
        dyn_cast<mlir::arith::ConstantIntOp>(firstDelay.getDefiningOp());
    auto secondDelayOp =
        dyn_cast<mlir::arith::ConstantIntOp>(secondDelay.getDefiningOp());

    auto mergeDelayValue = firstDelayOp.value() + secondDelayOp.value();

    auto mergeConstant = rewriter.create<mlir::arith::ConstantIntOp>(
        delayOp.getLoc(), mergeDelayValue, 32);

    // set first DelayOp duration to summed constant
    delayOp->setOperand(1, mergeConstant);

    // erase following delay
    rewriter.eraseOp(nextOp);

    return success();

  } // matchAndRewrite
};  // struct DelayAndDelayPattern
} // end anonymous namespace

void MergeDelayPass::runOnOperation() {

  Operation *operation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<DelayAndDelayPattern>(&getContext());

  mlir::GreedyRewriteConfig config;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  if (failed(
          applyPatternsAndFoldGreedily(operation, std::move(patterns), config)))
    signalPassFailure();

} // runOnOperation

llvm::StringRef MergeDelayPass::getArgument() const {
  return "pulse-merge-delay";
}

llvm::StringRef MergeDelayPass::getDescription() const {
  return "Merge sequencial delays on the same physical channel";
}

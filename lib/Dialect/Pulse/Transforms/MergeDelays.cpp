//===- MergeDelays.cpp - merges delays on the same target -------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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
    bool ignoreTarget = true;

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
    auto firstDelayPortOrFrame = delayOp.target();
    auto secondDelayPortOrFrame = nextDelayOp.target();

    if (!ignoreTarget && (firstDelayPortOrFrame != secondDelayPortOrFrame))
      return failure();

    // get delay times and sum as int 32
    // TODO: check to see if int 32 is sufficient
    auto firstDelay = delayOp.dur();
    auto secondDelay = nextDelayOp.dur();

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

  Operation *moduleOperation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.insert<DelayAndDelayPattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();

} // runOnOperation

llvm::StringRef MergeDelayPass::getArgument() const {
  return "pulse-merge-delay";
}

llvm::StringRef MergeDelayPass::getDescription() const {
  return "Merge sequencial delays on the same physical channel";
}

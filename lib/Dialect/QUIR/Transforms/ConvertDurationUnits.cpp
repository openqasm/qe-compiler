//===- ConvertDurationUnits.cpp - Convert Duration Unis  --------*- C++ -*-===//
//
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
///  This file implements the pass for converting the units of Durations
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/ConvertDurationUnits.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quir;


namespace {
    // This pattern matches on a BarrierOp follows by a CallCircuitOp separated by
    // non-quantum ops
    struct ConstantDurationPattern : public OpRewritePattern<BarrierOp> {
    explicit ConstantDurationPattern(MLIRContext *ctx)
        : OpRewritePattern<BarrierOp>(ctx) {}

    LogicalResult matchAndRewrite(BarrierOp barrierOp,
                                    PatternRewriter &rewriter) const override {

        // check for circuit op to merge with after moving barrier
        auto prevCallCircuitOp =
            prevQuantumOpOrNullOfType<CallCircuitOp>(barrierOp);
        if (!prevCallCircuitOp)
        return failure();

        auto callCircuitOp =
            getNextOpAndCompareOverlap<BarrierOp, CallCircuitOp>(barrierOp);
        if (!callCircuitOp.hasValue())
        return failure();

        barrierOp->moveAfter(callCircuitOp.getValue().getOperation());

        return success();
    } // matchAndRewrite
    };  // struct ConstantDurationPattern

} // anonymous namespace



// Entry point for the pass.
void QUIRConvertDurationUnitsPass::runOnOperation() {
    Operation *moduleOperation = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantDurationPattern>(&getContext());

    if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();

}

llvm::StringRef QUIRConvertDurationUnitsPass::getArgument() const {
  return "convert-quir-duration-units";
}
llvm::StringRef QUIRConvertDurationUnitsPass::getDescription() const {
  return "Convert the units of durations (and associated constant values) to the desired units";
}

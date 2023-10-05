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
    struct ConstantDurationPattern : public OpRewritePattern<ConstantOp> {
    explicit ConstantDurationPattern(MLIRContext *ctx)
        : OpRewritePattern<ConstantOp>(ctx) {}

    LogicalResult matchAndRewrite(ConstantOp constantOp,
                                    PatternRewriter &rewriter) const override {

        return failure();
    } // matchAndRewrite
    };  // struct ConstantDurationPattern

} // anonymous namespace



// Entry point for the pass.
void ConvertDurationUnitsPass::runOnOperation() {
    Operation *moduleOperation = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantDurationPattern>(&getContext());

    if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();

}

TimeUnits ConvertDurationUnitsPass::getTargetConvertUnits() const {
    return units;
}

double ConvertDurationUnitsPass::getDtDuration() {
    if (dtDuration < 0.) {
        llvm::errs() << "Supplied duration of " << dtDuration << "s is invalid \n";
        signalPassFailure();

    }
    return dtDuration;
}

llvm::StringRef ConvertDurationUnitsPass::getArgument() const {
  return "convert-quir-duration-units";
}
llvm::StringRef ConvertDurationUnitsPass::getDescription() const {
  return "Convert the units of durations (and associated constant values) to the desired units";
}

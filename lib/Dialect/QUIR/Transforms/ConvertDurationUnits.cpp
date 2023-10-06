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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quir;


namespace {

    /// Materialize quir.constant durations with an incorrect
    /// type to the specified type.
    struct MaterializeDurationUnitsConversionPattern
        : public OpConversionPattern<quir::ConstantOp> {
    explicit MaterializeDurationUnitsConversionPattern(MLIRContext *ctx,
                                                mlir::TypeConverter &typeConverter, TimeUnits convertUnits, llvm::Optional<double> dtDuration)
        : OpConversionPattern(typeConverter, ctx, /*benefit=*/1), convertUnits(convertUnits), dtDuration(dtDuration) {}

        LogicalResult
        matchAndRewrite(quir::ConstantOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {

            auto duration = DurationAttr::get(getContext(),
                          rewriter.getType<DurationType>(convertUnits),
                          llvm::APFloat(1.0));
            rewriter.replaceOpWithNewOp<quir::ConstantOp>(op, duration);

            return success();
        } // matchAndRewrite

        private:
            TimeUnits convertUnits;
            llvm::Optional<double> dtDuration;


    };  // struct MaterializeDurationUnitsConversionPattern

} // anonymous namespace


void ConvertDurationUnitsPass::runOnOperation() {
    Operation *moduleOperation = getOperation();

    // Extract conversion units
    auto units = getTargetConvertUnits();

    // Extract dt conversion factor if necessary
    llvm::Optional<double> dtConversion;
    if (units == TimeUnits::dt)
        dtConversion = getDtDuration();

    auto &context = getContext();
    ConversionTarget target(context);

    // Type converter to ensure only durations of target units exist
    // after cnoversion
    TypeConverter typeConverter;

    typeConverter.addConversion([&](quir::DurationType t) -> Optional<Type> {
                    if (t.getUnits() == units)
                        return t;
                    return DurationType::get(t.getContext(), units);
                });

    RewritePatternSet patterns(&context);

    // Only constant declared durations if their type is not
    // the target output duration type.
    target.addDynamicallyLegalOp<quir::ConstantOp>([&](quir::ConstantOp op) {
        auto type = op.getType().cast<DurationType>();
        if (!type)
            return true;

        auto convertUnits = type.getUnits();
        if (convertUnits == getTargetConvertUnits())
            return true;
        return false;
    });


    patterns.add<MaterializeDurationUnitsConversionPattern>(&getContext(), typeConverter, getTargetConvertUnits(), dtConversion);


    if(failed(applyPartialConversion(moduleOperation, target, std::move(patterns))))
        return signalPassFailure();
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

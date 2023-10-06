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


    template <typename T>
    Optional<Type> legalizeType(T t) {
        return t;
    }


    class DurationTypeConverter : public TypeConverter {

        public:
            DurationTypeConverter(TimeUnits convertUnits) {
                // Convert durations to the appropriate type
                addConversion([&](quir::DurationType t) -> Optional<Type> {
                    if (t.getUnits() == convertUnits)
                        return t;
                    return DurationType::get(t.getContext(), convertUnits);
                });
                addConversion(legalizeType<QubitType>);
            }
    }; // DurationTypeConverter

    /// Convert quir.constant durations with an incorrect
    /// type to the specified type.
    struct DurationUnitsConstantOpConversionPattern
        : public OpConversionPattern<quir::ConstantOp> {
    explicit DurationUnitsConstantOpConversionPattern(MLIRContext *ctx,
                                                DurationTypeConverter &typeConverter, llvm::Optional<double> dtDuration)
        : OpConversionPattern(typeConverter, ctx, /*benefit=*/1), dtDuration(dtDuration) {}

        LogicalResult
        matchAndRewrite(quir::ConstantOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {

            auto dstType = this->typeConverter->convertType(op.getType());
            if (!dstType)
                return failure();

            auto duration = DurationAttr::get(getContext(), dstType.cast<DurationType>(), llvm::APFloat(1.0));
            rewriter.replaceOpWithNewOp<quir::ConstantOp>(op, duration);

            return success();
        } // matchAndRewrite

        private:
            llvm::Optional<double> dtDuration;


    };  // struct DurationUnitsConstantOpConversionPattern


    /// Convert quir.delay duration with an incorrect
    /// type to the specified type.
    struct DurationUnitsDelayOpConversionPattern
        : public OpConversionPattern<quir::DelayOp> {
    explicit DurationUnitsDelayOpConversionPattern(MLIRContext *ctx,
                                                DurationTypeConverter &typeConverter)
        : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

        LogicalResult
        matchAndRewrite(quir::DelayOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {

            rewriter.updateRootInPlace(
                op, [&]() { op->setOperands(adaptor.getOperands());
            });

            return success();
        } // matchAndRewrite

    };  // struct DurationUnitsDelayOpConversionPattern


    struct DurationUnitsCircuitOpConversionPattern : public OpConversionPattern<quir::CircuitOp> {
        explicit DurationUnitsCircuitOpConversionPattern(MLIRContext *ctx, DurationTypeConverter &typeConverter)
            : OpConversionPattern(typeConverter, ctx, /*benefit=*/1), typeConverter(typeConverter) {}

        LogicalResult
        matchAndRewrite(quir::CircuitOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {

            auto funcLikeType = op.getType();
            DurationTypeConverter::SignatureConversion signatureConverter(
                funcLikeType.getNumInputs());
            auto newFuncLikeType = funcLikeType;
            if (!newFuncLikeType)
            return failure();

            // Create a new `CircuitOp`
            Location loc = op.getLoc();
            StringRef name = op.getName();
            auto newFuncLikeOp = rewriter.create<quir::CircuitOp>(loc, name, newFuncLikeType);

            rewriter.inlineRegionBefore(op.getBody(), newFuncLikeOp.getBody(),
                                        newFuncLikeOp.end());
            if (failed(rewriter.convertRegionTypes(&newFuncLikeOp.getBody(), typeConverter,
                                                &signatureConverter))) {
            return failure();
            }
            rewriter.eraseOp(op);
            return success();
        }

        DurationTypeConverter &typeConverter;

    }; // struct DurationUnitsCircuitOpConversionPattern

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
    DurationTypeConverter typeConverter(units);

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

    target.addDynamicallyLegalOp<quir::DelayOp>([&](quir::DelayOp op) {
        auto type = op.time().getType().cast<DurationType>();

        auto convertUnits = type.getUnits();
        if (convertUnits == getTargetConvertUnits())
            return true;
        return false;
    });


    target.addDynamicallyLegalOp<quir::CircuitOp>([&](quir::CircuitOp op) {
        for (auto type : op.getArgumentTypes()) {
            if (!type.isa<DurationType>())
                continue;
            auto durationType = type.cast<DurationType>();

            auto convertUnits = durationType.getUnits();
            if (convertUnits != getTargetConvertUnits())
                return false;
        }
        return true;
    });


    patterns.add<DurationUnitsConstantOpConversionPattern>(&getContext(), typeConverter, dtConversion);
    patterns.add<
        DurationUnitsDelayOpConversionPattern,
        DurationUnitsCircuitOpConversionPattern
    >(&getContext(), typeConverter);


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

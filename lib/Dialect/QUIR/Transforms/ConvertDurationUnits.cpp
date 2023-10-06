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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

            mlir::FunctionType convertFunctionSignature(FunctionType funcTy, DurationTypeConverter::SignatureConversion &result) {
                // Convert argument types one by one and check for errors.
                for (auto &en : llvm::enumerate(funcTy.getInputs())) {
                    Type type = en.value();
                    SmallVector<Type, 8> converted;
                    auto convertedType = convertType(type);
                    if (!convertedType)
                        convertedType = type;

                    converted.push_back(convertedType);
                    result.addInputs(en.index(), converted);
                }

                // Convert signature for block arguments
                SmallVector<Type, 8> argTypes;
                argTypes.reserve(llvm::size(result.getConvertedTypes()));
                for (Type type : result.getConvertedTypes())
                    argTypes.push_back(type);

                // Convert tyes for results as well
                SmallVector<Type, 1> resultTypes;
                for (auto type : funcTy.getResults()) {
                    auto convertedType = convertType(type);
                    if (!convertedType)
                        convertedType = type;
                    resultTypes.push_back(convertedType);
                }

                return mlir::FunctionType::get(funcTy.getContext(), argTypes, resultTypes);
            } // convertFunctionSignature

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

    /// Convert duration unit types via the adaptor with an incorrect
    /// type to the specified type.
    template <typename OperationType>
    struct DurationUnitsConversionPattern
        : public OpConversionPattern<OperationType> {
    explicit DurationUnitsConversionPattern(MLIRContext *ctx,
                                                DurationTypeConverter &typeConverter)
        : OpConversionPattern<OperationType>(typeConverter, ctx, /*benefit=*/1) {}

        LogicalResult
        matchAndRewrite(OperationType op, typename OperationType::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {

            rewriter.updateRootInPlace(
                op, [&]() { op->setOperands(adaptor.getOperands());
            });

            return success();
        } // matchAndRewrite

    };  // struct DurationUnitsConversionPattern


    template <typename FunctionType>
    struct DurationUnitsFunctionOpConversionPattern : public OpConversionPattern<FunctionType> {
        explicit DurationUnitsFunctionOpConversionPattern(MLIRContext *ctx, DurationTypeConverter &typeConverter)
            : OpConversionPattern<FunctionType>(typeConverter, ctx, /*benefit=*/1), typeConverter(typeConverter) {}

        LogicalResult
        matchAndRewrite(FunctionType funcLikeOp, typename FunctionType::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {

            auto funcLikeType = funcLikeOp.getType();
            DurationTypeConverter::SignatureConversion signatureConverter(
                funcLikeType.getNumInputs());
            auto newFuncLikeType = typeConverter.convertFunctionSignature(funcLikeOp.getType(), signatureConverter);
            if (!newFuncLikeType)
                return failure();

            // Create a new function operation
            Location loc = funcLikeOp.getLoc();
            StringRef name = funcLikeOp.getName();
            auto newFuncLikeOp = rewriter.create<FunctionType>(loc, name, newFuncLikeType);

            rewriter.inlineRegionBefore(funcLikeOp.getBody(), newFuncLikeOp.getBody(),
                                        newFuncLikeOp.end());
            if (failed(rewriter.convertRegionTypes(&newFuncLikeOp.getBody(), typeConverter,
                                                &signatureConverter)))
                return failure();


            rewriter.eraseOp(funcLikeOp);

            return success();
        }

        private:
            DurationTypeConverter &typeConverter;


    }; // struct DurationUnitsFunctionOpConversionPattern


    bool checkTypeNeedConversion(mlir::Type type, TimeUnits targetConvertUnits) {
        if (!type.isa<DurationType>())
            return false;
        auto durationType = type.cast<DurationType>();

        auto convertUnits = durationType.getUnits();
        if (convertUnits != targetConvertUnits)
            return true;
        return false;
    } // checkTypeNeedConversion


} // anonymous namespace


void ConvertDurationUnitsPass::runOnOperation() {
    Operation *moduleOperation = getOperation();

    // Extract conversion units
    auto targetConvertUnits = getTargetConvertUnits();

    // Extract dt conversion factor if necessary
    llvm::Optional<double> dtConversion;
    if (targetConvertUnits  == TimeUnits::dt)
        dtConversion = getDtDuration();

    auto &context = getContext();
    ConversionTarget target(context);

    // Type converter to ensure only durations of target units exist
    // after cnoversion
    DurationTypeConverter typeConverter(targetConvertUnits );

    RewritePatternSet patterns(&context);


    target.addDynamicallyLegalOp<quir::ConstantOp>([&](quir::ConstantOp op) {
        auto type = op.getType().cast<DurationType>();
        return !checkTypeNeedConversion(type, targetConvertUnits);
    });

    target.addDynamicallyLegalOp<quir::CircuitOp, mlir::FuncOp>([&](mlir::FunctionOpInterface op) {
        for (auto type : op.getArgumentTypes()) {
            if(checkTypeNeedConversion(type, targetConvertUnits))
                return false;
        }
        for (auto type : op.getResultTypes()) {
            if(checkTypeNeedConversion(type, targetConvertUnits))
                return false;
        }

        return true;
    });

    // Only constant declared durations if their type is not
    // the target output duration type.
    target.addDynamicallyLegalOp<
            quir::DelayOp,
            quir::ReturnOp,
            mlir::ReturnOp
        >([&](mlir::Operation *op) {
            for (auto type : op->getOperandTypes()) {
                if(checkTypeNeedConversion(type, targetConvertUnits))
                    return false;
            }
            return true;
    });


    patterns.add<DurationUnitsConstantOpConversionPattern>(&getContext(), typeConverter, dtConversion);
    patterns.add<
        DurationUnitsConversionPattern<quir::DelayOp>,
        DurationUnitsConversionPattern<quir::ReturnOp>,
        DurationUnitsFunctionOpConversionPattern<quir::CircuitOp>,
        DurationUnitsConversionPattern<mlir::ReturnOp>,
        DurationUnitsFunctionOpConversionPattern<mlir::FuncOp>
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
  return "Convert the units of all duration types within the module to the specified units";
}

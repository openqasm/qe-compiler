//===- ConvertDurationUnits.cpp - Convert Duration Units  -------*- C++ -*-===//
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

#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIREnums.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <sys/types.h>
#include <utility>

using namespace mlir;
using namespace mlir::quir;

namespace {

/// @brief  A custom unit type converter which marks
/// duration types that are not of the target unit as
/// illegal (to ensure they are converted) and all other types legal.
class DurationTypeConverter : public TypeConverter {

public:
  DurationTypeConverter(const TimeUnits convertUnits)
      : convertUnits_(convertUnits) {
    // Convert durations to the appropriate type
    addConversion([&](mlir::Type t) -> std::optional<Type> {
      // All non-durations are legal in the conversion.
      if (!t.isa<DurationType>())
        return t;

      auto duration = t.dyn_cast<DurationType>();
      if (duration.getUnits() == convertUnits_)
        return t;

      return DurationType::get(duration.getContext(), convertUnits_);
    });
  }

  /// Convert duration types for an input function signature returning a new
  /// function type signature as well as populating the SignatureConversion
  /// object for block region argument type mapping.
  mlir::FunctionType
  convertFunctionSignature(FunctionType funcTy,
                           DurationTypeConverter::SignatureConversion &result) {
    // Convert argument types one by one and check for errors.
    for (const auto &en : llvm::enumerate(funcTy.getInputs())) {
      Type const type = en.value();
      SmallVector<Type, 8> converted;
      auto convertedType = convertType(type);
      if (!convertedType)
        convertedType = type;

      converted.push_back(convertedType);
      // Load the signature indicies and what they have been converted to.
      result.addInputs(en.index(), converted);
    }

    // Convert types for results as well
    SmallVector<Type, 1> resultTypes;
    for (auto type : funcTy.getResults()) {
      auto convertedType = convertType(type);
      if (!convertedType)
        convertedType = type;
      resultTypes.push_back(convertedType);
    }

    return mlir::FunctionType::get(funcTy.getContext(),
                                   result.getConvertedTypes(), resultTypes);
  } // convertFunctionSignature

private:
  TimeUnits convertUnits_;

}; // DurationTypeConverter

/// Convert quir.constant durations with an incorrect
/// type to the specified type. This is the pattern
/// that actually computes the unit conversion.
struct DurationUnitsConstantOpConversionPattern
    : public OpConversionPattern<quir::ConstantOp> {
  explicit DurationUnitsConstantOpConversionPattern(
      MLIRContext *ctx, DurationTypeConverter &typeConverter, double dtTimestep)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
        dtTimestep(dtTimestep) {}

  LogicalResult
  matchAndRewrite(quir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto duration = op.getValue().dyn_cast<quir::DurationAttr>();
    if (!duration)
      return failure();

    auto dstType = this->typeConverter->convertType(op.getType());

    if (!dstType)
      return failure();

    auto units = dstType.cast<DurationType>().getUnits();

    DurationAttr const newDuration =
        duration.getConvertedDurationAttr(units, dtTimestep);
    rewriter.replaceOpWithNewOp<quir::ConstantOp>(op, newDuration);

    return success();
  } // matchAndRewrite

private:
  double dtTimestep;

}; // struct DurationUnitsConstantOpConversionPattern

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

    // Update the operand input types using the adaptor
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });

    return success();
  } // matchAndRewrite

}; // struct DurationUnitsConversionPattern

/// Convert duration unit types via the adaptor
/// for operations that return types.
template <typename ReturnsTypeOp>
struct DurationUnitsReturnsTypeOpConversionPattern
    : public OpConversionPattern<ReturnsTypeOp> {
  explicit DurationUnitsReturnsTypeOpConversionPattern(
      MLIRContext *ctx, DurationTypeConverter &typeConverter)
      : OpConversionPattern<ReturnsTypeOp>(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(ReturnsTypeOp op, typename ReturnsTypeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const auto numResults = op.getNumResults();
    SmallVector<Type, 1> resultTypes;
    for (uint i = 0; i < numResults; ++i) {
      auto type = op.getType(i);
      auto convertedType = this->typeConverter->convertType(type);
      if (!convertedType)
        convertedType = type;
      resultTypes.push_back(convertedType);
    }
    rewriter.replaceOpWithNewOp<ReturnsTypeOp>(
        op, resultTypes, adaptor.getOperands(), op->getAttrs());

    return success();
  } // matchAndRewrite

}; // struct DurationUnitsReturnsTypeOpConversionPattern

/// Update the types of operations that implement the callable interface.
/// Care must be taken to properly map the types of containing regions
/// using the SignatureConversion inteface. These are not well documented
/// but this implementation follows the SPIRVToLLVM implementation.
template <typename FunctionType>
struct DurationUnitsFunctionOpConversionPattern
    : public OpConversionPattern<FunctionType> {
  explicit DurationUnitsFunctionOpConversionPattern(
      MLIRContext *ctx, DurationTypeConverter &typeConverter)
      : OpConversionPattern<FunctionType>(typeConverter, ctx, /*benefit=*/1),
        typeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(FunctionType funcLikeOp,
                  typename FunctionType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto funcLikeType = funcLikeOp.getFunctionType();
    // Create a signature converter for our interface
    DurationTypeConverter::SignatureConversion signatureConverter(
        funcLikeType.getNumInputs());

    // Generate the updated function signature type.
    // The signatureConverter is built up which will then be
    // used below to map region types in the rewriter.
    auto newFuncLikeType = typeConverter.convertFunctionSignature(
        funcLikeOp.getFunctionType(), signatureConverter);
    if (!newFuncLikeType)
      return failure();

    // Create a new function operation with the update signature type.
    Location const loc = funcLikeOp.getLoc();
    StringRef const name = funcLikeOp.getName();
    auto newFuncLikeOp =
        rewriter.create<FunctionType>(loc, name, newFuncLikeType);

    // Update the internal Regions/Blocks while ensuring the updated types are
    // appropriately propagated using the SignatureConverter we built up.
    rewriter.inlineRegionBefore(funcLikeOp.getBody(), newFuncLikeOp.getBody(),
                                newFuncLikeOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncLikeOp.getBody(),
                                           typeConverter, &signatureConverter)))
      return failure();

    rewriter.eraseOp(funcLikeOp);

    return success();
  }

private:
  DurationTypeConverter &typeConverter;

}; // struct DurationUnitsFunctionOpConversionPattern

/// Helper method to check that the duration units need to be converted.
bool checkTypeNeedsConversion(mlir::Type type, TimeUnits targetConvertUnits) {
  if (!type.isa<DurationType>())
    return false;
  auto durationType = type.cast<DurationType>();

  auto convertUnits = durationType.getUnits();
  if (convertUnits != targetConvertUnits)
    return true;
  return false;
} // checkTypeNeedsConversion

} // anonymous namespace

void ConvertDurationUnitsPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  // Extract conversion units
  auto targetConvertUnits = getTargetConvertUnits();

  double const dtConversion = getDtTimestep();

  auto &context = getContext();
  ConversionTarget target(context);

  // Type converter to ensure only durations of target units exist
  // after conversion
  DurationTypeConverter typeConverter(targetConvertUnits);

  RewritePatternSet patterns(&context);

  // Patterns below ensure that all operations that might have
  // durations are marked for potential unit conversion.
  target.addDynamicallyLegalOp<quir::ConstantOp>([&](quir::ConstantOp op) {
    auto type = op.getType();
    if (type.isa<DurationType>())
      return !checkTypeNeedsConversion(type, targetConvertUnits);

    return true;
  });

  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
    for (auto type : op.getArgumentTypes())
      if (checkTypeNeedsConversion(type, targetConvertUnits))
        return false;
    for (auto type : op.getResultTypes())
      if (checkTypeNeedsConversion(type, targetConvertUnits))
        return false;

    return true;
  });

  target.addDynamicallyLegalOp<quir::CircuitOp>([&](quir::CircuitOp op) {
    for (auto type : op.getArgumentTypes())
      if (checkTypeNeedsConversion(type, targetConvertUnits))
        return false;
    for (auto type : op.getResultTypes())
      if (checkTypeNeedsConversion(type, targetConvertUnits))
        return false;

    return true;
  });

  // Only constant declared durations if their type is not
  // the target output duration type.
  target
      .addDynamicallyLegalOp<quir::DelayOp, quir::ReturnOp, quir::CallCircuitOp,
                             mlir::func::ReturnOp, mlir::func::CallOp>(
          [&](mlir::Operation *op) {
            for (auto type : op->getOperandTypes())
              if (checkTypeNeedsConversion(type, targetConvertUnits))
                return false;
            for (auto type : op->getResultTypes())
              if (checkTypeNeedsConversion(type, targetConvertUnits))
                return false;
            return true;
          });

  // Patterns to convert operations who's types might contain invalid duration
  // units.
  patterns.add<DurationUnitsConstantOpConversionPattern>(
      &getContext(), typeConverter, dtConversion);
  patterns.add<DurationUnitsConversionPattern<quir::DelayOp>,
               DurationUnitsConversionPattern<quir::ReturnOp>,
               DurationUnitsReturnsTypeOpConversionPattern<quir::CallCircuitOp>,
               DurationUnitsFunctionOpConversionPattern<quir::CircuitOp>,
               DurationUnitsConversionPattern<mlir::func::ReturnOp>,
               DurationUnitsReturnsTypeOpConversionPattern<mlir::func::CallOp>,
               DurationUnitsFunctionOpConversionPattern<mlir::func::FuncOp>>(
      &getContext(), typeConverter);

  if (failed(
          applyPartialConversion(moduleOperation, target, std::move(patterns))))
    return signalPassFailure();
}

TimeUnits ConvertDurationUnitsPass::getTargetConvertUnits() const {
  return units;
}

double ConvertDurationUnitsPass::getDtTimestep() { return dtTimestep; }

llvm::StringRef ConvertDurationUnitsPass::getArgument() const {
  return "convert-quir-duration-units";
}
llvm::StringRef ConvertDurationUnitsPass::getDescription() const {
  return "Convert the units of all duration types within the module to the "
         "specified units.";
}

llvm::StringRef ConvertDurationUnitsPass::getName() const {
  return "Convert Duration Units Pass (" + getArgument().str() + ")";
}

//===- CBitOperations.cpp --------------------------------------*- C++ -*-===//
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
/// \file
/// This file implements patterns for lowering operations on cbits.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/Passes.h"

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::quir {

template <class QuirOp, class StdOp>
struct BinaryCBitCBitOpConversionPattern : public OpConversionPattern<QuirOp> {
  explicit BinaryCBitCBitOpConversionPattern(MLIRContext *ctx,
                                             mlir::TypeConverter &typeConverter)
      : OpConversionPattern<QuirOp>(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(QuirOp bitOp, typename QuirOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();

    assert(operands.size() == 2 &&
           "expect binary cbit operation to have two operands");

    for (Value operand : bitOp->getOperands()) {
      auto operandType = operand.getType();
      assert((operandType.isa<mlir::quir::CBitType>() ||
              operandType.isInteger(1)) &&
             "Binary CBit CBit Op pattern operand must be cbit or i1");
      if (!(operandType.isa<mlir::quir::CBitType>() ||
            operandType.isInteger(1)))
        return failure();
    }

    for (Value operand : operands)
      if (!operand.getType().isSignlessInteger())
        return failure();

    mlir::Type operandType = bitOp.lhs().getType();
    if (auto cbitType = operandType.dyn_cast<mlir::quir::CBitType>())
      if (cbitType.getWidth() > 64)
        return failure();
    // TODO support more than 64 cbits

    auto stdOp = rewriter.create<StdOp>(bitOp.getLoc(), operands[0].getType(),
                                        operands[0], operands[1]);
    rewriter.replaceOp(bitOp, {stdOp.getResult()});
    return success();
  }
};

struct CbitAssignBitOpConversionPattern
    : public OpConversionPattern<AssignCbitBitOp> {
  explicit CbitAssignBitOpConversionPattern(MLIRContext *ctx,
                                            mlir::TypeConverter &typeConverter)
      : OpConversionPattern<AssignCbitBitOp>(typeConverter, ctx,
                                             /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(AssignCbitBitOp bitOp, AssignCbitBitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto const location = bitOp.getLoc();
    auto cbitWidth = bitOp.cbit_width();

    // just a single bit? then generate a value assign
    if (cbitWidth == 1) {

      rewriter.create<mlir::quir::VariableAssignOp>(
          location, bitOp.variable_nameAttr(), adaptor.assigned_bit());

      rewriter.replaceOp(bitOp, mlir::ValueRange({}));
      return success();
    }

    if (cbitWidth.ugt(64)) // the IndexAttr is turned into an llvm::APInt
      return failure();

    auto oldRegisterValue = rewriter.create<mlir::quir::UseVariableOp>(
        location,
        /*rewriter.getType<mlir::quir::CBitType>(cbitWidth.getZExtValue())*/
        rewriter.getIntegerType(cbitWidth.getZExtValue()),
        bitOp.variable_name());

    auto registerWithInsertedBit =
        rewriter.create<mlir::quir::Cbit_InsertBitOp>(
            location, oldRegisterValue.getType(), oldRegisterValue,
            adaptor.assigned_bit(), adaptor.indexAttr());

    rewriter.create<mlir::quir::VariableAssignOp>(
        location, bitOp.variable_nameAttr(), registerWithInsertedBit);
    rewriter.replaceOp(bitOp, mlir::ValueRange({}));
    return success();
  }
};

static int getCbitOrIntBitWidth(mlir::Type t) {
  assert((t.isa<quir::CBitType>() || t.isSignlessInteger()) &&
         "expect CBitType or integer type");

  if (auto cbt = t.dyn_cast<quir::CBitType>())
    return cbt.getWidth();

  if (auto intType = t.dyn_cast<mlir::IntegerType>())
    return intType.getWidth();

  llvm::report_fatal_error("unhandled type");
}

struct CbitInsertBitOpConversionPattern
    : public OpConversionPattern<Cbit_InsertBitOp> {
  explicit CbitInsertBitOpConversionPattern(MLIRContext *ctx,
                                            mlir::TypeConverter &typeConverter)
      : OpConversionPattern<Cbit_InsertBitOp>(typeConverter, ctx,
                                              /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(Cbit_InsertBitOp bitOp, Cbit_InsertBitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto const location = bitOp.getLoc();
    auto cbitWidth = getCbitOrIntBitWidth(adaptor.operand().getType());

    // just a single bit? then replace whole bitmap
    if (cbitWidth == 1) {
      rewriter.replaceOp(bitOp, mlir::ValueRange({adaptor.assigned_bit()}));
      return success();
    }

    if (cbitWidth > 64)
      return failure();

    uint64_t mask = ~((1ull) << bitOp.index().getZExtValue());
    auto maskOp =
        rewriter.create<mlir::arith::ConstantIntOp>(location, mask, cbitWidth);

    auto maskedBitmap = rewriter.create<mlir::LLVM::AndOp>(
        location, maskOp.getType(), adaptor.operand(), maskOp);

    mlir::Value extendedBit = rewriter.create<mlir::LLVM::ZExtOp>(
        location, maskOp.getType(), adaptor.assigned_bit());

    if (!bitOp.index().isNonPositive()) {
      auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
          location, extendedBit.getType(),
          rewriter.getIntegerAttr(extendedBit.getType(),
                                  bitOp.index().trunc(cbitWidth)));
      extendedBit = rewriter.create<mlir::LLVM::ShlOp>(location, extendedBit,
                                                       shiftAmount);
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(bitOp, maskOp.getType(),
                                                  maskedBitmap, extendedBit);
    return success();
  }
};

struct CbitExtractBitOpConversionPattern
    : public OpConversionPattern<Cbit_ExtractBitOp> {
  explicit CbitExtractBitOpConversionPattern(MLIRContext *ctx,
                                             mlir::TypeConverter &typeConverter)
      : OpConversionPattern<Cbit_ExtractBitOp>(typeConverter, ctx,
                                               /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(Cbit_ExtractBitOp bitOp, Cbit_ExtractBitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto const location = bitOp.getLoc();
    auto convertedOperand = adaptor.operand();

    assert((bitOp.operand().getType().isa<quir::CBitType>() ||
            bitOp.operand().getType().isSignlessInteger()) &&

           "expect operand of CBitType for quir.cbit_extractbit");

    auto bitWidth = getCbitOrIntBitWidth(bitOp.operand().getType());

    if (bitWidth == 1) {
      // just pass-through single bit values
      rewriter.replaceOp(bitOp, {convertedOperand});
      return success();
    }

    auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
        location, convertedOperand.getType(),
        rewriter.getIntegerAttr(convertedOperand.getType(),
                                bitOp.index().trunc(bitWidth)));
    auto shiftedRegister = rewriter.create<mlir::LLVM::LShrOp>(
        location, convertedOperand.getType(), convertedOperand, shiftAmount);
    auto extractedBit = rewriter.create<mlir::LLVM::TruncOp>(
        location, bitOp.getType(), shiftedRegister);
    rewriter.replaceOp(bitOp, {extractedBit});

    return success();
  }
};

void populateCBitOperationsPatterns(RewritePatternSet &patterns,
                                    mlir::TypeConverter &typeConverter,
                                    bool includeBitmapOperationPatterns) {
  auto *context = patterns.getContext();
  assert(context);

  patterns.insert<BinaryCBitCBitOpConversionPattern<mlir::quir::Cbit_AndOp,
                                                    mlir::LLVM::AndOp>>(
      context, typeConverter);
  patterns.insert<BinaryCBitCBitOpConversionPattern<mlir::quir::Cbit_OrOp,
                                                    mlir::LLVM::OrOp>>(
      context, typeConverter);
  patterns.insert<BinaryCBitCBitOpConversionPattern<mlir::quir::Cbit_XorOp,
                                                    mlir::LLVM::XOrOp>>(
      context, typeConverter);
  patterns.insert<CbitAssignBitOpConversionPattern>(context, typeConverter);

  if (includeBitmapOperationPatterns) {
    patterns.insert<CbitExtractBitOpConversionPattern>(context, typeConverter);
    patterns.insert<CbitInsertBitOpConversionPattern>(context, typeConverter);
  }
}

}; // namespace mlir::quir

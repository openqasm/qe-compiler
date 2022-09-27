//===- CBitOperations.cpp --------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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
      assert(operand.getType().isa<mlir::quir::CBitType>() &&
             "Binary CBit CBit Op pattern operand must be cbit");
      if (!operand.getType().isa<mlir::quir::CBitType>())
        return failure();
    }

    for (Value operand : operands)
      if (!operand.getType().isSignlessInteger())
        return failure();

    mlir::Type operandType = bitOp.lhs().getType();
    auto cbitType = operandType.dyn_cast<mlir::quir::CBitType>();

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
  matchAndRewrite(AssignCbitBitOp bitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();

    assert(operands.size() == 1 &&
           "quir.assign_cbit_bit should have a single operand");
    auto const location = bitOp.getLoc();
    auto cbitWidth = bitOp.cbit_width();

    // just a single bit? then generate a value assign
    if (cbitWidth == 1) {

      rewriter.create<mlir::quir::VariableAssignOp>(
          location, bitOp.variable_nameAttr(), operands[0]);

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
    uint64_t mask = ~((1ull) << bitOp.index().getZExtValue());
    auto maskOp = rewriter.create<mlir::arith::ConstantIntOp>(
        location, mask, cbitWidth.getZExtValue());
    auto maskedRegisterValue = rewriter.create<mlir::LLVM::AndOp>(
        location, maskOp.getType(), oldRegisterValue, maskOp);
    mlir::Value extendedBit = rewriter.create<mlir::LLVM::ZExtOp>(
        location, maskOp.getType(), operands[0]);

    if (!bitOp.index().isNonPositive()) {

      auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
          location, extendedBit.getType(),
          rewriter.getIntegerAttr(
              extendedBit.getType(),
              bitOp.index().trunc(cbitWidth.getZExtValue())));
      extendedBit = rewriter.create<mlir::LLVM::ShlOp>(location, extendedBit,
                                                       shiftAmount);
    }

    auto registerWithInsertedBit = rewriter.create<mlir::LLVM::OrOp>(
        location, maskOp.getType(), maskedRegisterValue, extendedBit);

    rewriter.create<mlir::quir::VariableAssignOp>(
        location, bitOp.variable_nameAttr(), registerWithInsertedBit);
    rewriter.replaceOp(bitOp, mlir::ValueRange({}));
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
  matchAndRewrite(Cbit_ExtractBitOp bitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto const location = bitOp.getLoc();
    auto operands = adaptor.getOperands();

    assert(operands.size() == 1 &&
           "quir.cbit_extractbit should have a single (converted) operand");
    assert(bitOp.operand().getType().isa<quir::CBitType>() &&
           "expect operand of CBitType for quir.cbit_extractbit");

    auto bitWidth =
        bitOp.operand().getType().dyn_cast<quir::CBitType>().getWidth();

    if (bitWidth == 1) {
      // just pass-through single bit values
      rewriter.replaceOp(bitOp, {operands[0]});
      return success();
    }

    auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
        location, operands[0].getType(),
        rewriter.getIntegerAttr(operands[0].getType(),
                                bitOp.index().trunc(bitWidth)));
    auto shiftedRegister = rewriter.create<mlir::LLVM::LShrOp>(
        location, operands[0].getType(), operands[0], shiftAmount);
    auto extractedBit = rewriter.create<mlir::LLVM::TruncOp>(
        location, bitOp.getType(), shiftedRegister);
    rewriter.replaceOp(bitOp, {extractedBit});

    return success();
  }
};

void populateCBitOperationsPatterns(RewritePatternSet &patterns,
                                    mlir::TypeConverter &typeConverter) {
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
  patterns.insert<CbitExtractBitOpConversionPattern>(context, typeConverter);
}

}; // namespace mlir::quir

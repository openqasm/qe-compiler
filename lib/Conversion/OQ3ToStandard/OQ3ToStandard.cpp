//===- OQ3ToStandard.cpp - OpenQASM 3 to Standard patterns ------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file implements patterns to convert OpenQASM 3 to Standard dialect.
///
//===----------------------------------------------------------------------===//

#include "Conversion/OQ3ToStandard/OQ3ToStandard.h"
#include "Dialect/QUIR/Transforms/Passes.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
// #include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::oq3 {
template <class OQ3Op, class StdOp>
struct CBitBinaryOpConversionPattern : public OpConversionPattern<OQ3Op> {
  using OpConversionPattern<OQ3Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OQ3Op op, typename OQ3Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    assert(operands.size() == 2 &&
           "Expect binary CBit operation to have exactly two operands.");

    for (Value operand : op->getOperands()) {
      auto operandType = operand.getType();
      assert((operandType.isa<quir::CBitType>() || operandType.isInteger(1)) &&
             "Binary CBit operation pattern operands must be `CBit` or i1");
      if (!(operandType.isa<quir::CBitType>() || operandType.isInteger(1)))
        return failure();
    }

    for (Value operand : operands)
      if (!operand.getType().isSignlessInteger())
        return failure();

    Type operandType = op.lhs().getType();
    if (auto cbitType = operandType.dyn_cast<quir::CBitType>())
      if (cbitType.getWidth() > 64)
        return failure();
    // TODO: Support > 64 `CBits`

    auto stdOp = rewriter.create<StdOp>(op.getLoc(), operands[0].getType(),
                                        operands[0], operands[1]);
    rewriter.replaceOp(op, {stdOp.getResult()});
    return success();
  }
};

struct CBitAssignBitOpConversionPattern
    : public OpConversionPattern<AssignCBitBitOp> {
  using OpConversionPattern<AssignCBitBitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssignCBitBitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto const loc = op.getLoc();
    auto cbitWidth = op.cbit_width();

    // just a single bit? then generate a value assign
    if (cbitWidth == 1) {

      rewriter.create<mlir::oq3::AssignVariableOp>(loc, op.variable_nameAttr(),
                                                   adaptor.assigned_bit());

      rewriter.replaceOp(op, mlir::ValueRange({}));
      return success();
    }

    if (cbitWidth.ugt(64)) // the IndexAttr is turned into an llvm::APInt
      return failure();

    auto oldRegisterValue = rewriter.create<mlir::oq3::UseVariableOp>(
        loc,
        /*rewriter.getType<mlir::quir::CBitType>(cbitWidth.getZExtValue())*/
        rewriter.getIntegerType(cbitWidth.getZExtValue()), op.variable_name());

    auto registerWithInsertedBit = rewriter.create<mlir::oq3::CBitInsertBitOp>(
        loc, oldRegisterValue.getType(), oldRegisterValue,
        adaptor.assigned_bit(), adaptor.indexAttr());

    rewriter.create<mlir::oq3::AssignVariableOp>(loc, op.variable_nameAttr(),
                                                 registerWithInsertedBit);
    rewriter.replaceOp(op, mlir::ValueRange({}));
    return success();
  }
};

static int getCBitOrIntBitWidth(mlir::Type t) {
  assert((t.isa<quir::CBitType>() || t.isSignlessInteger()) &&
         "expect CBitType or integer type");

  if (auto cbt = t.dyn_cast<quir::CBitType>())
    return cbt.getWidth();

  if (auto intType = t.dyn_cast<mlir::IntegerType>())
    return intType.getWidth();

  llvm::report_fatal_error("unhandled type");
}

struct CBitInsertBitOpConversionPattern
    : public OpConversionPattern<CBitInsertBitOp> {
  using OpConversionPattern<CBitInsertBitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CBitInsertBitOp op, CBitInsertBitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto const loc = op.getLoc();
    auto cbitWidth = getCBitOrIntBitWidth(adaptor.operand().getType());

    // just a single bit? then replace whole bitmap
    if (cbitWidth == 1) {
      rewriter.replaceOp(op, mlir::ValueRange({adaptor.assigned_bit()}));
      return success();
    }

    if (cbitWidth > 64)
      return failure();

    uint64_t mask = ~((1ull) << op.index().getZExtValue());
    auto maskOp =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, mask, cbitWidth);

    auto maskedBitmap = rewriter.create<mlir::LLVM::AndOp>(
        loc, maskOp.getType(), adaptor.operand(), maskOp);

    mlir::Value extendedBit = rewriter.create<mlir::LLVM::ZExtOp>(
        loc, maskOp.getType(), adaptor.assigned_bit());

    if (!op.index().isNonPositive()) {
      auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
          loc, extendedBit.getType(),
          rewriter.getIntegerAttr(extendedBit.getType(),
                                  op.index().trunc(cbitWidth)));
      extendedBit =
          rewriter.create<mlir::LLVM::ShlOp>(loc, extendedBit, shiftAmount);
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, maskOp.getType(),
                                                  maskedBitmap, extendedBit);
    return success();
  }
};

struct CBitExtractBitOpConversionPattern
    : public OpConversionPattern<CBitExtractBitOp> {
  using OpConversionPattern<CBitExtractBitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CBitExtractBitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto const location = op.getLoc();
    auto convertedOperand = adaptor.operand();

    assert((op.operand().getType().isa<quir::CBitType>() ||
            op.operand().getType().isSignlessInteger()) &&

           "expect operand of CBitType for oq3.cbit_extractbit");

    auto bitWidth = getCBitOrIntBitWidth(op.operand().getType());

    if (bitWidth == 1) {
      // just pass-through single bit values
      rewriter.replaceOp(op, {convertedOperand});
      return success();
    }

    auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
        location, convertedOperand.getType(),
        rewriter.getIntegerAttr(convertedOperand.getType(),
                                op.index().trunc(bitWidth)));
    auto shiftedRegister = rewriter.create<mlir::LLVM::LShrOp>(
        location, convertedOperand.getType(), convertedOperand, shiftAmount);
    auto extractedBit = rewriter.create<mlir::LLVM::TruncOp>(
        location, op.getType(), shiftedRegister);
    rewriter.replaceOp(op, {extractedBit});

    return success();
  }
};

void populateOQ3ToStandardConversionPatterns(
    RewritePatternSet &patterns, bool includeBitmapOperationPatterns) {
  // clang-format off
  patterns.add<
      CBitBinaryOpConversionPattern<oq3::CBitAndOp,
                                    LLVM::AndOp>,
      CBitBinaryOpConversionPattern<oq3::CBitOrOp,
                                    LLVM::OrOp>,
      CBitBinaryOpConversionPattern<oq3::CBitXorOp,
                                    LLVM::XOrOp>,
      CBitAssignBitOpConversionPattern>(patterns.getContext());

  if (includeBitmapOperationPatterns) {
    patterns.add<
        CBitExtractBitOpConversionPattern,
        CBitInsertBitOpConversionPattern>(patterns.getContext());
  }
  // clang-format on
}

}; // namespace mlir::oq3

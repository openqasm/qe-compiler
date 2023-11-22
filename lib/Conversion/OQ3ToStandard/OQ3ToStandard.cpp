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
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Transforms/Passes.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace oq3;
using namespace quir;

//===----------------------------------------------------------------------===//
// CBitOp conversion
//===----------------------------------------------------------------------===//
template <class OQ3Op, class StdOp>
struct CBitBinaryOpConversionPattern : public OQ3ToStandardConversion<OQ3Op> {
  using OQ3ToStandardConversion<OQ3Op>::OQ3ToStandardConversion;

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
    : public OQ3ToStandardConversion<CBitAssignBitOp> {
  using OQ3ToStandardConversion<CBitAssignBitOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CBitAssignBitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto const loc = op.getLoc();
    auto cbitWidth = op.getCbitWidth();

    // just a single bit? then generate a value assign
    if (cbitWidth == 1) {

      rewriter.create<mlir::oq3::VariableAssignOp>(loc, op.getVariableNameAttr(),
                                                   adaptor.getAssignedBit());

      rewriter.replaceOp(op, mlir::ValueRange({}));
      return success();
    }

    if (cbitWidth.ugt(64)) // the IndexAttr is turned into an llvm::APInt
      return failure();

    auto oldRegisterValue = rewriter.create<mlir::oq3::VariableLoadOp>(
        loc, rewriter.getIntegerType(cbitWidth.getZExtValue()),
        op.getVariableName());

    auto registerWithInsertedBit = rewriter.create<mlir::oq3::CBitInsertBitOp>(
        loc, oldRegisterValue.getType(), oldRegisterValue,
        adaptor.getAssignedBit(), adaptor.getIndexAttr());

    rewriter.create<mlir::oq3::VariableAssignOp>(loc, op.getVariableNameAttr(),
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
    : public OQ3ToStandardConversion<CBitInsertBitOp> {
  using OQ3ToStandardConversion<CBitInsertBitOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CBitInsertBitOp op, CBitInsertBitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto const loc = op.getLoc();
    auto cbitWidth = getCBitOrIntBitWidth(adaptor.getOperand().getType());

    // just a single bit? then replace whole bitmap
    if (cbitWidth == 1) {
      rewriter.replaceOp(op, mlir::ValueRange({adaptor.getAssignedBit()}));
      return success();
    }

    if (cbitWidth > 64)
      return failure();

    uint64_t mask = ~((1ull) << op.getIndex().getZExtValue());
    auto maskOp =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, mask, cbitWidth);

    auto maskedBitmap = rewriter.create<mlir::LLVM::AndOp>(
        loc, maskOp.getType(), adaptor.getOperand(), maskOp);

    mlir::Value extendedBit = rewriter.create<mlir::LLVM::ZExtOp>(
        loc, maskOp.getType(), adaptor.getAssignedBit());

    if (!op.getIndex().isNonPositive()) {
      APInt truncated = op.getIndex();
      if (static_cast<uint>(cbitWidth) < truncated.getBitWidth())
        truncated = truncated.trunc(cbitWidth);
      auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
          loc, extendedBit.getType(),
          rewriter.getIntegerAttr(extendedBit.getType(), truncated));
      extendedBit =
          rewriter.create<mlir::LLVM::ShlOp>(loc, extendedBit, shiftAmount);
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, maskOp.getType(),
                                                  maskedBitmap, extendedBit);
    return success();
  }
};

struct CBitExtractBitOpConversionPattern
    : public OQ3ToStandardConversion<CBitExtractBitOp> {
  using OQ3ToStandardConversion<CBitExtractBitOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CBitExtractBitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto const location = op.getLoc();
    auto convertedOperand = adaptor.getOperand();

    assert((op.getOperand().getType().isa<quir::CBitType>() ||
            op.getOperand().getType().isSignlessInteger()) &&

           "expect operand of CBitType for oq3.cbit_extractbit");

    auto bitWidth = getCBitOrIntBitWidth(op.getOperand().getType());

    if (bitWidth == 1) {
      // just pass-through single bit values
      rewriter.replaceOp(op, {convertedOperand});
      return success();
    }

    APInt truncated = op.getIndex();
    if (static_cast<uint>(bitWidth) < truncated.getBitWidth())
      truncated = truncated.trunc(bitWidth);
    auto shiftAmount = rewriter.create<mlir::arith::ConstantOp>(
        location, convertedOperand.getType(),
        rewriter.getIntegerAttr(convertedOperand.getType(), truncated));
    auto shiftedRegister = rewriter.create<mlir::LLVM::LShrOp>(
        location, convertedOperand.getType(), convertedOperand, shiftAmount);
    auto extractedBit = rewriter.create<mlir::LLVM::TruncOp>(
        location, op.getType(), shiftedRegister);
    rewriter.replaceOp(op, {extractedBit});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cast conversion
//===----------------------------------------------------------------------===//
namespace {
/// @brief Pattern for converting cast ops that produce bools from
/// integers
struct CastIntToBoolConversionPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult match(CastOp op) const override {
    if (!isBoolType(op.getType()))
      return failure();

    auto argType = op.getArg().getType();

    if (argType.isIntOrIndex())
      return success();

    return failure();
  } // match

  void rewrite(CastOp op, CastOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {

    auto argType = op.getArg().getType();

    // per OpenQASM3 spec, cast from int to bool by comparing val != 0
    auto constInt0Op = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), argType, rewriter.getIntegerAttr(argType, 0));
    auto cmpOp = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::ne, op.getArg(),
        constInt0Op.getResult());
    rewriter.replaceOp(op, ValueRange{cmpOp});
  }
};

/// @brief Struct for converting cast ops that produce integers from cbits
struct CastCBitToIntConversionPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getArg().getType().isa<CBitType>())
      return failure();
    if (!op.getOut().getType().isIntOrIndex())
      return failure();

    auto cbitType = op.getArg().getType().dyn_cast<CBitType>();
    auto outWidth = op.getOut().getType().getIntOrFloatBitWidth();

    if (cbitType.getWidth() > outWidth)
      // cannot reinterpret without losing bits!
      return failure();

    if (cbitType.getWidth() < outWidth) {
      // need to zero-extend to match output type width
      rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, op.getOut().getType(),
                                                        adaptor.getArg());
      return success();
    }

    // 1:1 conversion of cbit (which is lowered to int) to int
    assert(op.getOut().getType() == adaptor.getArg().getType() &&
           "cbit lowers to int");
    rewriter.replaceOp(op, adaptor.getArg());
    return success();
  } // matchAndRewrite
};  // struct CastCBitToIntConversionPattern

/// @brief Conversion pattern for cast ops that produce cbits from integers
struct CastIntToCBitConversionPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getArg().getType().isIntOrIndexOrFloat())
      return failure();
    if (!op.getOut().getType().isa<CBitType>())
      return failure();

    auto cbitType = op.getOut().getType().dyn_cast<CBitType>();

    // assign single bit from an integer
    if (cbitType.getWidth() == 1) {
      auto truncateOp = rewriter.create<mlir::LLVM::TruncOp>(
          op.getLoc(), rewriter.getI1Type(), adaptor.getArg());

      rewriter.replaceOp(op, mlir::ValueRange{truncateOp});
      return success();
    }
    if (op.getArg().getType().getIntOrFloatBitWidth() == cbitType.getWidth()) {
      // 1:1 conversion of int to cbit
      if (cbitType.getWidth() > 64)
        return failure();

      rewriter.replaceOp(op, adaptor.getArg());
      return success();
    }

    return failure();
  } // matchAndRewrite
};  // struct CastIntToCBitConversionPattern

/// @brief Conversion pattern for cast ops that produce integers from index
/// values
struct CastIndexToIntPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // check if the input is index type
    if (!op.getArg().getType().isIndex())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(
        op, op.getOut().getType(), adaptor.getArg());
    return success();
  } // matchAndRewrite
};  // struct CastIndexToIntPattern

/// @brief Conversion pattern that drops CastOps that have been made redundant
/// by type conversion (e.g., cbit<1> -> i1)
struct RemoveConvertedNilCastsPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getType() != adaptor.getArg().getType())
      return failure();

    rewriter.replaceOp(op, {adaptor.getArg()});
    return success();
  } // matchAndRewrite

}; // struct RemoveConvertedNilCastsPattern

struct CastFromFloatConstPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto constOp = op.getArg().getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp)
      return failure();

    auto floatAttr = constOp.getValue().dyn_cast<mlir::FloatAttr>();
    if (!floatAttr)
      return failure();

    rewriter.replaceOp(op, {adaptor.getArg()});
    return success();
  } // CastFromFloatConstPattern

}; // struct RemoveConvertedNilCastsPattern

struct RemoveI1ToCBitCastsPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getArg().getType() != rewriter.getI1Type())
      return failure();
    auto cbitType = op.getType().dyn_cast<CBitType>();
    if (!cbitType || cbitType.getWidth() != 1)
      return failure();

    rewriter.replaceOp(op, {adaptor.getArg()});
    return success();
  } // matchAndRewrite

}; // struct RemoveI1ToCBitCastsPattern

struct WideningIntCastsPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getOperand().getType().isSignlessInteger())
      return failure();
    if (!op.getType().isSignlessInteger())
      return failure();

    assert(op.getOperand().getType() == adaptor.getArg().getType() &&
           "unexpected type conversion for built-in integer types");

    if (op.getOperand().getType().getIntOrFloatBitWidth() >=
        op.getType().getIntOrFloatBitWidth())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, op.getType(),
                                                      adaptor.getArg());
    return success();
  } // matchAndRewrite
};  // struct WideningIntCastsPattern
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//
void oq3::populateOQ3ToStandardConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    bool includeBitmapOperationPatterns) {
  // clang-format off
  // CBit ops
  patterns.add<
      CBitBinaryOpConversionPattern<oq3::CBitAndOp,
                                    LLVM::AndOp>,
      CBitBinaryOpConversionPattern<oq3::CBitOrOp,
                                    LLVM::OrOp>,
      CBitBinaryOpConversionPattern<oq3::CBitXorOp,
                                    LLVM::XOrOp>,
      CBitAssignBitOpConversionPattern>(patterns.getContext(), typeConverter);

  if (includeBitmapOperationPatterns) {
    patterns.add<
        CBitExtractBitOpConversionPattern,
        CBitInsertBitOpConversionPattern>(patterns.getContext(), typeConverter);
  }

  // Cast ops
  patterns.add<
      CastCBitToIntConversionPattern,
      CastIntToCBitConversionPattern,
      CastIntToBoolConversionPattern,
      CastIndexToIntPattern,
      RemoveConvertedNilCastsPattern,
      RemoveI1ToCBitCastsPattern,
      CastFromFloatConstPattern,
      WideningIntCastsPattern>(patterns.getContext(), typeConverter);
  // clang-format on
}

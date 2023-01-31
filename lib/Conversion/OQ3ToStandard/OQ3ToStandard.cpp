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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
        loc, rewriter.getIntegerType(cbitWidth.getZExtValue()),
        op.variable_name());

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
    : public OQ3ToStandardConversion<CBitInsertBitOp> {
  using OQ3ToStandardConversion<CBitInsertBitOp>::OQ3ToStandardConversion;

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
    : public OQ3ToStandardConversion<CBitExtractBitOp> {
  using OQ3ToStandardConversion<CBitExtractBitOp>::OQ3ToStandardConversion;

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

//===----------------------------------------------------------------------===//
// Cast conversion
//===----------------------------------------------------------------------===//
namespace {
/// @brief Pattern for converting cast ops that produce bools from
/// integers
struct CastIntegerToBoolConversionPattern
    : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult match(CastOp op) const override {
    if (!isBoolType(op.getType()))
      return failure();

    auto argType = op.arg().getType();

    if (argType.isIntOrIndex())
      return success();

    return failure();
  } // match

  void rewrite(CastOp op, CastOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {

    auto argType = op.arg().getType();

    // per OpenQASM3 spec, cast from int to bool by comparing val != 0
    auto constInt0Op = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), argType, rewriter.getIntegerAttr(argType, 0));
    auto cmpOp = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::ne, op.arg(),
        constInt0Op.getResult());
    rewriter.replaceOp(op, ValueRange{cmpOp});
  }
};

/// @brief Struct for converting cast ops that produce integers from cbits
struct CastCBitToIntConversionPat : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.arg().getType().isa<CBitType>())
      return failure();
    if (!op.out().getType().isIntOrIndex())
      return failure();

    auto cbitType = op.arg().getType().dyn_cast<CBitType>();
    auto outWidth = op.out().getType().getIntOrFloatBitWidth();

    if (cbitType.getWidth() > outWidth)
      // cannot reinterpret without losing bits!
      return failure();

    if (cbitType.getWidth() < outWidth) {
      // need to zero-extend to match output type width
      rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, op.out().getType(),
                                                        adaptor.arg());
      return success();
    }

    // 1:1 conversion of cbit (which is lowered to int) to int
    assert(op.out().getType() == adaptor.arg().getType() &&
           "cbit lowers to int");
    rewriter.replaceOp(op, adaptor.arg());
    return success();
  } // matchAndRewrite
};  // struct CastCBitToIntConversionPat

/// @brief Conversion pattern for cast ops that produce cbits from integers
struct CastIntToCBitConversionPat : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.arg().getType().isIntOrIndexOrFloat())
      return failure();
    if (!op.out().getType().isa<CBitType>())
      return failure();

    auto cbitType = op.out().getType().dyn_cast<CBitType>();

    // assign single bit from an integer
    if (cbitType.getWidth() == 1) {
      auto truncateOp = rewriter.create<mlir::LLVM::TruncOp>(
          op.getLoc(), rewriter.getI1Type(), adaptor.arg());

      rewriter.replaceOp(op, mlir::ValueRange{truncateOp});
      return success();
    }
    if (op.arg().getType().getIntOrFloatBitWidth() == cbitType.getWidth()) {
      // 1:1 conversion of int to cbit
      if (cbitType.getWidth() > 64)
        return failure();

      rewriter.replaceOp(op, adaptor.arg());
      return success();
    }

    return failure();
  } // matchAndRewrite
};  // struct CastIntToCBitConversionPat

/// @brief Conversion pattern for cast ops that produce integers from index
/// values
struct CastIndexToIntegerPat : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // check if the input is index type
    if (!op.arg().getType().isIndex())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(
        op, op.out().getType(), adaptor.arg());
    return success();
  } // matchAndRewrite
};  // struct CastIndexToIntegerPat

/// @brief Conversion pattern that drops CastOps that have been made redundant
/// by type conversion (e.g., cbit<1> -> i1)
struct RemoveConvertedNilCastsPat : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getType() != adaptor.arg().getType())
      return failure();

    rewriter.replaceOp(op, {adaptor.arg()});
    return success();
  } // matchAndRewrite

}; // struct RemoveConvertedNilCastsPat

struct RemoveI1ToCBitCastsPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.arg().getType() != rewriter.getI1Type())
      return failure();
    auto cbitType = op.getType().dyn_cast<CBitType>();
    if (!cbitType || cbitType.getWidth() != 1)
      return failure();

    rewriter.replaceOp(op, {adaptor.arg()});
    return success();
  } // matchAndRewrite

}; // struct RemoveI1ToCBitCastsPattern

struct WideningIntegerCastsPattern : public OQ3ToStandardConversion<CastOp> {
  using OQ3ToStandardConversion<CastOp>::OQ3ToStandardConversion;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getOperand().getType().isSignlessInteger())
      return failure();
    if (!op.getType().isSignlessInteger())
      return failure();

    assert(op.getOperand().getType() == adaptor.arg().getType() &&
           "unexpected type conversion for built-in integer types");

    if (op.getOperand().getType().getIntOrFloatBitWidth() >=
        op.getType().getIntOrFloatBitWidth())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, op.getType(),
                                                      adaptor.arg());
    return success();
  } // matchAndRewrite
};  // struct WideningIntegerCastsPattern
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
  // patterns.add<CastCBitToIntConversionPat,
  //     CastIntToCBitConversionPat,
  //     CastIntegerToBoolConversionPattern,
  //     CastIndexToIntegerPat,
  //     RemoveConvertedNilCastsPat,
  //     RemoveI1ToCBitCastsPattern,
  //     WideningIntegerCastsPattern>(patterns.getContext(), typeConverter);
  // clang-format on
}

void oq3::populateOQ3ToStandardCastConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  // clang-format off
  // Cast ops
  patterns.add<CastCBitToIntConversionPat,
      CastIntToCBitConversionPat,
      CastIntegerToBoolConversionPattern,
      CastIndexToIntegerPat,
      RemoveConvertedNilCastsPat,
      RemoveI1ToCBitCastsPattern,
      WideningIntegerCastsPattern>(patterns.getContext(), typeConverter);
  // clang-format on
}

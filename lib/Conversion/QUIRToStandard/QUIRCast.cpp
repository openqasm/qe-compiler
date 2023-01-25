//===- QUIRCast.cpp - Convert cast op to Std --------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  This file implements the pass for converting the QUIR cast op
///  to the std dialect
///
//===----------------------------------------------------------------------===//

#include "Conversion/QUIRToStandard/QUIRCast.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::quir;

namespace {

/// @brief Pattern for converting cast ops that produce bools from integers
struct CastIntegerToBoolConversionPattern
    : public OpConversionPattern<quir::CastOp> {
  explicit CastIntegerToBoolConversionPattern(
      MLIRContext *ctx, mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /* benefit= */ 1) {}

  LogicalResult match(CastOp castOp) const override {
    if (!isBoolType(castOp.getType()))
      return failure();

    auto argType = castOp.arg().getType();

    if (argType.isIntOrIndex())
      return success();

    return failure();
  } // match

  void rewrite(CastOp castOp, CastOp::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {

    auto argType = castOp.arg().getType();

    // per OpenQASM3 spec, cast from int to bool by comparing val != 0
    auto constInt0Op = rewriter.create<mlir::arith::ConstantOp>(
        castOp.getLoc(), argType, rewriter.getIntegerAttr(argType, 0));
    auto cmpOp = rewriter.create<mlir::LLVM::ICmpOp>(
        castOp.getLoc(), mlir::LLVM::ICmpPredicate::ne, castOp.arg(),
        constInt0Op.getResult());
    rewriter.replaceOp(castOp, ValueRange{cmpOp});
  }
};

/// @brief Struct for converting cast ops that produce integers from cbits
struct CastCBitToIntConversionPat : public OpConversionPattern<CastOp> {
  explicit CastCBitToIntConversionPat(MLIRContext *ctx,
                                      mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!castOp.arg().getType().isa<CBitType>())
      return failure();
    if (!castOp.out().getType().isIntOrIndex())
      return failure();

    auto cbitType = castOp.arg().getType().dyn_cast<CBitType>();
    auto outWidth = castOp.out().getType().getIntOrFloatBitWidth();

    if (cbitType.getWidth() > outWidth)
      // cannot reinterpret without losing bits!
      return failure();

    if (cbitType.getWidth() < outWidth) {
      // need to zero-extend to match output type width
      rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(
          castOp, castOp.out().getType(), adaptor.arg());
      return success();
    }

    // 1:1 conversion of cbit (which is lowered to int) to int
    assert(castOp.out().getType() == adaptor.arg().getType() &&
           "cbit lowers to int");
    rewriter.replaceOp(castOp, adaptor.arg());
    return success();
  } // matchAndRewrite
};  // struct CastCBitToIntConversionPat

/// @brief Conversion pattern for cast ops that produce cbits from integers
struct CastIntToCBitConversionPat : public OpConversionPattern<CastOp> {
  explicit CastIntToCBitConversionPat(MLIRContext *ctx,
                                      mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!castOp.arg().getType().isIntOrIndexOrFloat())
      return failure();
    if (!castOp.out().getType().isa<CBitType>())
      return failure();

    auto cbitType = castOp.out().getType().dyn_cast<CBitType>();

    // assign single bit from an integer
    if (cbitType.getWidth() == 1) {
      auto truncateOp = rewriter.create<mlir::LLVM::TruncOp>(
          castOp.getLoc(), rewriter.getI1Type(), adaptor.arg());

      rewriter.replaceOp(castOp, mlir::ValueRange{truncateOp});
      return success();
    }
    if (castOp.arg().getType().getIntOrFloatBitWidth() == cbitType.getWidth()) {
      // 1:1 conversion of int to cbit
      if (cbitType.getWidth() > 64)
        return failure();

      rewriter.replaceOp(castOp, adaptor.arg());
      return success();
    }

    return failure();
  } // matchAndRewrite
};  // struct CastIntToCBitConversionPat

/// @brief Conversion pattern for cast ops that produce integers from index
/// values
struct CastIndexToIntegerPat : public OpConversionPattern<CastOp> {
  explicit CastIndexToIntegerPat(MLIRContext *ctx,
                                 mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // check if the input is index type
    if (!castOp.arg().getType().isIndex())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(
        castOp, castOp.out().getType(), adaptor.arg());
    return success();
  } // matchAndRewrite
};  // struct CastIndexToIntegerPat

/// @brief Conversion pattern that drops CastOps that have been made redundant
/// by type conversion (e.g., cbit<1> -> i1)
struct RemoveConvertedNilCastsPat : public OpConversionPattern<CastOp> {
  explicit RemoveConvertedNilCastsPat(MLIRContext *ctx,
                                      mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (castOp.getType() != adaptor.arg().getType())
      return failure();

    rewriter.replaceOp(castOp, {adaptor.arg()});
    return success();
  } // matchAndRewrite

}; // struct RemoveConvertedNilCastsPat

struct RemoveI1ToCBitCastsPattern : public OpConversionPattern<CastOp> {
  explicit RemoveI1ToCBitCastsPattern(MLIRContext *ctx,
                                      mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.arg().getType() != rewriter.getI1Type())
      return failure();
    auto cbitType = castOp.getType().dyn_cast<CBitType>();
    if (!cbitType || cbitType.getWidth() != 1)
      return failure();

    rewriter.replaceOp(castOp, {adaptor.arg()});
    return success();
  } // matchAndRewrite

}; // struct RemoveI1ToCBitCastsPattern

struct WideningIntegerCastsPattern : public OpConversionPattern<quir::CastOp> {
  explicit WideningIntegerCastsPattern(MLIRContext *ctx,
                                       mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(quir::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!castOp.getOperand().getType().isSignlessInteger())
      return failure();
    if (!castOp.getType().isSignlessInteger())
      return failure();

    assert(castOp.getOperand().getType() == adaptor.arg().getType() &&
           "unexpected type conversion for built-in integer types");

    if (castOp.getOperand().getType().getIntOrFloatBitWidth() >=
        castOp.getType().getIntOrFloatBitWidth())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(castOp, castOp.getType(),
                                                      adaptor.arg());
    return success();
  } // matchAndRewrite
};  // struct WideningIntegerCastsPattern

} // end anonymous namespace

void mlir::quir::populateQUIRCastPatterns(RewritePatternSet &patterns,
                                          mlir::TypeConverter &typeConverter) {
  auto *ctx = patterns.getContext();
  assert(ctx);

  patterns.insert<CastCBitToIntConversionPat>(ctx, typeConverter);
  patterns.insert<CastIntToCBitConversionPat>(ctx, typeConverter);
  patterns.insert<CastIntegerToBoolConversionPattern>(ctx, typeConverter);
  patterns.insert<CastIndexToIntegerPat>(ctx, typeConverter);
  patterns.insert<RemoveConvertedNilCastsPat>(ctx, typeConverter);
  patterns.insert<RemoveI1ToCBitCastsPattern>(ctx, typeConverter);
  patterns.insert<WideningIntegerCastsPattern>(ctx, typeConverter);
}

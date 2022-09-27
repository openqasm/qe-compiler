//===- QUIRToStd.cpp - Convert QUIR to Std Dialect --------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file implements passes for converting QUIR to std dialect
//
//===----------------------------------------------------------------------===//

#include "Conversion/QUIRToStandard/QUIRToStandard.h"

#include "MockUtils.h"

#include "Conversion/QUIRToStandard/TypeConversion.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace qssc::targets::mock::conversion {
struct ReturnConversionPat : public OpConversionPattern<mlir::ReturnOp> {

  explicit ReturnConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(mlir::ReturnOp retOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    rewriter.create<mlir::ReturnOp>(retOp->getLoc(), operands);
    rewriter.replaceOp(retOp, {});
    return success();
  } // matchAndRewrite
};  // struct ReturnConversionPat

// Convert quir.constant op to a std dialect constant op
// convert angles to integer values so we don't have to use soft-float
struct ConstantOpConversionPat : public OpConversionPattern<quir::ConstantOp> {

  explicit ConstantOpConversionPat(MLIRContext *ctx,
                                   TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(quir::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto angleAttr = constOp.value().dyn_cast<quir::AngleAttr>()) {
      auto angleWidth =
          constOp.getType().dyn_cast<quir::AngleType>().getWidth();

      // cannot handle non-parameterized angle types
      if (!angleWidth.hasValue())
        return failure();

      int64_t multiplier = (int64_t)1 << (int64_t)angleWidth.getValue();
      // shift the floating point value up by the desired precision
      double fVal = angleAttr.getValue().convertToDouble() * multiplier;
      IntegerType iType;
      if (angleWidth.getValue() > 31)
        iType = rewriter.getI64Type();
      else
        iType = rewriter.getI32Type();
      IntegerAttr iAttr = rewriter.getIntegerAttr(iType, (int64_t)fVal);

      auto arithConstOp = rewriter.create<mlir::arith::ConstantOp>(
          constOp->getLoc(), iType, iAttr);
      rewriter.replaceOp(constOp, {arithConstOp});
      return success();
    }
    // attribute type is not handled (for now)
    return failure();
  } // matchAndRewrite
};  // struct ConstantOpConversionPat

template <class QuirOp, class StdOp>
struct AngleBinOpConversionPat : public OpConversionPattern<QuirOp> {

  explicit AngleBinOpConversionPat(MLIRContext *ctx,
                                   TypeConverter &typeConverter)
      : OpConversionPattern<QuirOp>(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(QuirOp binOp, typename QuirOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto angleWidth =
        binOp.lhs().getType().template dyn_cast<quir::AngleType>().getWidth();

    // cannot handle non-parameterized angle types
    if (!angleWidth.hasValue())
      return failure();

    int64_t maskVal = ((int64_t)1 << (int64_t)angleWidth.getValue()) - 1;
    auto iType = operands[0].getType().template dyn_cast<IntegerType>();
    IntegerAttr iAttr = rewriter.getIntegerAttr(iType, maskVal);

    auto stdOp =
        rewriter.create<StdOp>(binOp.getLoc(), iType, operands[0], operands[1]);
    auto maskOp =
        rewriter.create<mlir::arith::ConstantOp>(binOp->getLoc(), iType, iAttr);
    auto andOp = rewriter.create<mlir::LLVM::AndOp>(
        binOp->getLoc(), stdOp.getResult(), maskOp.getResult());
    rewriter.replaceOp(binOp, {andOp.getODSResults(0)});
    return success();
  } // matchAndRewrite
};  // struct AngleBinOpConversionPat

// convert the comm op to a value to be removed or remove it completely as it is
// not supported by the mock target.
template <class CommOp>
struct CommOpConversionPat : public OpConversionPattern<CommOp> {

  explicit CommOpConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
      : OpConversionPattern<CommOp>(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CommOp commOp, typename CommOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto numResults = commOp.getOperation()->getNumResults();

    switch (numResults) {
    case 0:
      rewriter.replaceOp(commOp.getOperation(), {});
      return success();

    case 1: {
      // shift the floating point value up by the desired precision
      int64_t iVal = 1;
      IntegerType i1Type = rewriter.getI1Type();

      IntegerAttr iAttr = rewriter.getIntegerAttr(i1Type, iVal);
      auto constOp = rewriter.create<mlir::arith::ConstantOp>(commOp->getLoc(),
                                                              i1Type, iAttr);
      rewriter.replaceOp(commOp.getOperation(), {constOp.getODSResults(0)});
      return success();
    }

    default:
      commOp.emitOpError()
          << "Error: " << commOp.getOperation() << " with " << numResults
          << " is not currently handled by the pattern CommOpConversionPat!\n";
      return failure();
    }
  } // matchAndRewrite
};  // struct CommOpConversionPat

void MockQUIRToStdPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  // attempt to apply the conversion only to the controller module
  ModuleOp controllerModuleOp = getControllerModule(moduleOp);
  if (!controllerModuleOp)
    controllerModuleOp = moduleOp;

  QuirTypeConverter typeConverter;
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect, scf::SCFDialect,
                         arith::ArithmeticDialect, LLVM::LLVMDialect,
                         quir::QUIRDialect>();
  target.addIllegalOp<quir::RecvOp, quir::BroadcastOp>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
    return typeConverter.isSignatureLegal(op.getCalleeType());
  });
  target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp op) {
    return typeConverter.isLegal(op.getOperandTypes());
  });

  auto *context = &getContext();
  RewritePatternSet patterns(context);
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  patterns.insert<ConstantOpConversionPat>(context, typeConverter);

  // Replace Receive ops with constants to be optimized away.
  patterns.insert<CommOpConversionPat<quir::RecvOp>>(context, typeConverter);
  patterns.insert<CommOpConversionPat<quir::BroadcastOp>>(context,
                                                          typeConverter);
  patterns.insert<ReturnConversionPat>(context, typeConverter);
  patterns
      .insert<AngleBinOpConversionPat<quir::Angle_AddOp, mlir::arith::AddIOp>>(
          context, typeConverter);
  patterns
      .insert<AngleBinOpConversionPat<quir::Angle_SubOp, mlir::arith::SubIOp>>(
          context, typeConverter);
  patterns
      .insert<AngleBinOpConversionPat<quir::Angle_MulOp, mlir::arith::MulIOp>>(
          context, typeConverter);
  patterns
      .insert<AngleBinOpConversionPat<quir::Angle_DivOp, mlir::arith::DivSIOp>>(
          context, typeConverter);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(controllerModuleOp, target,
                                    std::move(patterns)))) {
  }
  // If we fail conversion remove remaining ops for the Mock target.
  controllerModuleOp.walk([&](Operation *op) {
    if (llvm::isa<quir::QUIRDialect>(op->getDialect())) {
      llvm::outs() << "Removing unsupported " << op->getName() << " \n";
      op->dropAllReferences();
      op->dropAllDefinedValueUses();
      op->erase();
    }
  });

} // QUIRToStdPass::runOnOperation()

llvm::StringRef MockQUIRToStdPass::getArgument() const {
  return "mock-quir-to-std";
}

llvm::StringRef MockQUIRToStdPass::getDescription() const {
  return "Convert QUIR ops to std dialect";
}

} // namespace qssc::targets::mock::conversion

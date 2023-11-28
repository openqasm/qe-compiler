//===- QUIRToStd.cpp - Convert QUIR to Std Dialect --------------*- C++ -*-===//
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
//
//  This file implements passes for converting QUIR to std dialect
//
//===----------------------------------------------------------------------===//
#include "Conversion/QUIRToStandard/QUIRToStandard.h"
#include "Conversion/QUIRToStandard/TypeConversion.h"
#include "Conversion/QUIRToStandard/VariablesToGlobalMemRefConversion.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "MockUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace qssc::targets::systems::mock::conversion {
struct ReturnConversionPat : public OpConversionPattern<mlir::func::ReturnOp> {

  explicit ReturnConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(mlir::func::ReturnOp retOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<mlir::func::ReturnOp>(retOp->getLoc(),
                                          adaptor.getOperands());
    rewriter.eraseOp(retOp);
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
    if (auto angleAttr = constOp.getValue().dyn_cast<quir::AngleAttr>()) {
      auto angleWidth = constOp.getType().cast<quir::AngleType>().getWidth();

      // cannot handle non-parameterized angle types
      if (!angleWidth.has_value())
        return failure();

      int64_t multiplier = (int64_t)1 << (int64_t)angleWidth.value();
      // shift the floating point value up by the desired precision
      double fVal = angleAttr.getValue().convertToDouble() * multiplier;
      IntegerType iType;
      if (angleWidth.value() > 31)
        iType = rewriter.getI64Type();
      else
        iType = rewriter.getI32Type();
      IntegerAttr iAttr = rewriter.getIntegerAttr(iType, (int64_t)fVal);

      auto arithConstOp = rewriter.create<mlir::arith::ConstantOp>(
          constOp->getLoc(), iType, iAttr);
      rewriter.replaceOp(constOp, arithConstOp);
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
    auto angleWidth = binOp.getLhs()
                          .getType()
                          .template dyn_cast<quir::AngleType>()
                          .getWidth();

    // cannot handle non-parameterized angle types
    if (!angleWidth.has_value())
      return failure();

    int64_t maskVal = ((int64_t)1 << (int64_t)angleWidth.value()) - 1;
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
      rewriter.eraseOp(commOp.getOperation());
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

void conversion::MockQUIRToStdPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect, arith::ArithDialect>();
}

void MockQUIRToStdPass::runOnOperation(MockSystem &system) {
  ModuleOp moduleOp = getOperation();

  // Attempt to apply the conversion only to the controller module
  ModuleOp controllerModuleOp = getControllerModule(moduleOp);
  if (!controllerModuleOp)
    controllerModuleOp = moduleOp;

  // First remove all arguments from synchronization ops
  controllerModuleOp->walk([](qcs::SynchronizeOp synchOp) {
    synchOp.getQubitsMutable().assign(ValueRange({}));
  });

  QuirTypeConverter typeConverter;
  auto *context = &getContext();
  ConversionTarget target(*context);

  target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect,
                         mlir::affine::AffineDialect, memref::MemRefDialect,
                         scf::SCFDialect, mlir::func::FuncDialect,
                         mlir::pulse::PulseDialect>();
  // Since we are converting QUIR -> STD/LLVM, make QUIR illegal.
  // Further, because OQ3 and QCS ops are migrated from QUIR, make them also
  // illegal.
  target
      .addIllegalDialect<quir::QUIRDialect, qcs::QCSDialect, oq3::OQ3Dialect>();
  target.addIllegalOp<qcs::RecvOp, qcs::BroadcastOp>();
  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return typeConverter.isSignatureLegal(op.getCalleeType());
  });
  target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) {
        return typeConverter.isLegal(op.getOperandTypes());
      });
  // We mark `ConstantOp` legal so we don't err when attempting to convert a
  // constant `DurationType`. (Only `AngleType` is currently handled by the
  // conversion pattern, `ConstantOpConversionPat`.)
  // Note that marking 'legal' does not preclude conversion if a pattern is
  // matched. However, because other ops also do not have implemented
  // conversions, we will still observe errors of the type:
  // ```
  // loc("-":0:0): error: failed to legalize operation 'namespace.op' that was
  // explicitly marked illegal
  // ```
  // which for the `mock_target` are harmless.
  target.addLegalOp<quir::ConstantOp>();
  target.addLegalOp<quir::SwitchOp>();
  target.addLegalOp<quir::YieldOp>();

  RewritePatternSet patterns(context);
  populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  // clang-format off
  patterns.add<ConstantOpConversionPat,
               ReturnConversionPat,
               CommOpConversionPat<qcs::RecvOp>,
               CommOpConversionPat<qcs::BroadcastOp>,
               AngleBinOpConversionPat<oq3::AngleAddOp, mlir::arith::AddIOp>,
               AngleBinOpConversionPat<oq3::AngleSubOp, mlir::arith::SubIOp>,
               AngleBinOpConversionPat<oq3::AngleMulOp, mlir::arith::MulIOp>,
               AngleBinOpConversionPat<oq3::AngleDivOp, mlir::arith::DivSIOp>>(
      context, typeConverter);
  // clang-format on

  quir::populateVariableToGlobalMemRefConversionPatterns(
      patterns, typeConverter, externalizeOutputVariables);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(controllerModuleOp, target,
                                    std::move(patterns)))) {
    // If we fail conversion remove remaining ops for the Mock target.
    controllerModuleOp.walk([&](Operation *op) {
      if (llvm::isa<oq3::OQ3Dialect>(op->getDialect()) ||
          llvm::isa<quir::QUIRDialect>(op->getDialect()) ||
          llvm::isa<qcs::QCSDialect>(op->getDialect())) {
        llvm::outs() << "Removing unsupported " << op->getName() << " \n";
        op->dropAllReferences();
        op->dropAllDefinedValueUses();
        op->erase();
      }
    });
  }
} // QUIRToStdPass::runOnOperation()

llvm::StringRef MockQUIRToStdPass::getArgument() const {
  return "mock-quir-to-std";
}

llvm::StringRef MockQUIRToStdPass::getDescription() const {
  return "Convert QUIR ops to std dialect";
}

} // namespace qssc::targets::systems::mock::conversion

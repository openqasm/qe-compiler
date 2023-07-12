//===- OQ3ToAER.cpp - Convert OQ3 to AER --------------*- C++ -*-===//
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
//  This file implements passes for converting OQ3 to AER
//
//===----------------------------------------------------------------------===//
#include "Conversion/OQ3ToAER.h"
#include "Conversion/QUIRToStandard/TypeConversion.h"
#include "Conversion/QUIRToStandard/VariablesToGlobalMemRefConversion.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "SimulatorUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

#include <exception>

namespace qssc::targets::simulator::conversion {

namespace {

static std::map<std::string, LLVM::LLVMFuncOp> aerFuncTable;

}

struct RemoveQCSInitConversionPat : public OpConversionPattern<qcs::SystemInitOp> {
  explicit RemoveQCSInitConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
    : OpConversionPattern(typeConverter, ctx, /* benefit= */1)
  {}
  
  LogicalResult matchAndRewrite(qcs::SystemInitOp initOp,
                                qcs::SystemInitOp::Adaptor adapter,
                                ConversionPatternRewriter &rewriter) const override {
    initOp.erase();
    return success();
  }
};

struct RemoveQCSShotInitConversionPat : public OpConversionPattern<qcs::ShotInitOp> {
  explicit RemoveQCSShotInitConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
    : OpConversionPattern(typeConverter, ctx, /* benefit= */1)
  {}
  
  LogicalResult matchAndRewrite(qcs::ShotInitOp initOp,
                                qcs::ShotInitOp::Adaptor adapter,
                                ConversionPatternRewriter &rewriter) const override {
    initOp.erase(); // TODO: Why rewriter.eraseOp(initOp) doen't work?
    return success();
  }
};

struct FinalizeConversionPat : public OpConversionPattern<qcs::SystemFinalizeOp> {
  explicit FinalizeConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, Value aerState)
    : OpConversionPattern(typeConverter, ctx, /* benefit= */1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(qcs::SystemFinalizeOp finOp,
                                qcs::SystemFinalizeOp::Adaptor adapter,
                                ConversionPatternRewriter &rewriter) const override {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointAfter(finOp);
    rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(),
                                  aerFuncTable.at("aer_state_finalize"),
                                  aerState);
    rewriter.eraseOp(finOp);
    return success();
  }
  
private:
  Value aerState;
};

struct QubitDeclConversionPat : public OpConversionPattern<quir::DeclareQubitOp> {
  explicit QubitDeclConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, Value aerState)
    : OpConversionPattern(typeConverter, ctx, 1),
      aerState(aerState)
  {}

  LogicalResult matchAndRewrite(quir::DeclareQubitOp declOp,
                                quir::DeclareQubitOp::Adaptor adapter,
                                ConversionPatternRewriter &rewriter) const override {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    const int width = declOp.getType().dyn_cast<quir::QubitType>().getWidth();
    rewriter.setInsertionPointAfter(declOp);
    const auto widthAttr = rewriter.getIntegerAttr(rewriter.getI64Type(), width);
    auto constOp = rewriter.create<arith::ConstantOp>(declOp->getLoc(),
                                                      rewriter.getI64Type(),
                                                      widthAttr);
    rewriter.create<LLVM::CallOp>(declOp->getLoc(),
                                  aerFuncTable.at("aer_allocate_qubits"),
                                  ValueRange{aerState, constOp});
    // TODO: erase
    //rewriter.eraseOp(declOp);
    return success();
  }
  
private:
  Value aerState;
};

struct BuiltinUConversionPat : public OpConversionPattern<quir::Builtin_UOp> {
  explicit BuiltinUConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, Value aerState)
    : OpConversionPattern(typeConverter, ctx, 1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(quir::Builtin_UOp op,
                                quir::Builtin_UOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    // TODO: erase
    //rewriter.eraseOp(op);
    op.getOperand(0).print(llvm::outs()); llvm::outs() << "\n";
    throw std::runtime_error("");
    return success();
  }
  
private:
  Value aerState;
};

struct BuiltinCXConversionPat : public OpConversionPattern<quir::BuiltinCXOp> {
  explicit BuiltinCXConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, Value aerState)
    : OpConversionPattern(typeConverter, ctx, 1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(quir::BuiltinCXOp op,
                                quir::BuiltinCXOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    // TODO: erase
    //rewriter.eraseOp(op);
    throw std::runtime_error("");
    return success();
  }
  
private:
  Value aerState;
};

struct MeasureConversionPat : public OpConversionPattern<quir::MeasureOp> {
  explicit MeasureConversionPat(MLIRContext *ctx, TypeConverter &typeConverter, Value aerState)
    : OpConversionPattern(typeConverter, ctx, 1),
      aerState(aerState)
  {}
  
  LogicalResult matchAndRewrite(quir::MeasureOp op,
                                quir::MeasureOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    // TODO: erase
    //rewriter.eraseOp(op);
    throw std::runtime_error("");
    return success();
  }
  
private:
  Value aerState;
};

struct AngleConversionPat : public OpConversionPattern<quir::ConstantOp> {
  explicit AngleConversionPat(MLIRContext *ctx, TypeConverter &typeConverter)
    : OpConversionPattern(typeConverter, ctx, 1)
  {}
  
  LogicalResult matchAndRewrite(quir::ConstantOp op,
                               quir::ConstantOp::Adaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override
  {
    if(auto angleAttr = op.value().dyn_cast<quir::AngleAttr>()) {
      rewriter.setInsertionPointAfter(op);
      const auto angle = angleAttr.getValue().convertToDouble();
      const auto fType = rewriter.getF64Type();
      FloatAttr fAttr = rewriter.getFloatAttr(fType, angle);
      rewriter.create<arith::ConstantOp>(op->getLoc(), fType, fAttr);
      // TODO: replace with above
      //rewriter.eraseOp(op);
    }
    return success();
  }
};


void conversion::QUIRToAERPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::AffineDialect, arith::ArithmeticDialect>();
}

void QUIRToAERPass::runOnOperation(SimulatorSystem &system) {
  ModuleOp moduleOp = getOperation();

  // First remove all arguments from synchronization ops
  moduleOp->walk([](qcs::SynchronizeOp synchOp) {
    synchOp.qubitsMutable().assign(ValueRange({}));
  });

  QuirTypeConverter typeConverter;
  auto *context = &getContext();
  ConversionTarget target(*context);

  target.addLegalDialect<arith::ArithmeticDialect, LLVM::LLVMDialect,
                         mlir::AffineDialect, memref::MemRefDialect,
                         scf::SCFDialect, StandardOpsDialect,
                         mlir::pulse::PulseDialect>();
  // Since we are converting QUIR -> AER/LLVM, make QUIR illegal.
  // Further, because OQ3 and QCS ops are migrated from QUIR, make them also
  // illegal.
  target.addIllegalDialect<qcs::QCSDialect>();

  RewritePatternSet patterns(context);
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<RemoveQCSInitConversionPat,
               RemoveQCSShotInitConversionPat,
               AngleConversionPat>(
      context, typeConverter);
  
  // Aer initialization
  declareAerFunctions(moduleOp);
  auto mainFunc = mlir::quir::getMainFunction(moduleOp);
  if(!mainFunc) return;

  OpBuilder builder(mainFunc);
  auto mainBody = &mainFunc->getRegion(0).getBlocks().front();
  builder.setInsertionPointToStart(mainBody);
  auto aerState = builder.create<LLVM::CallOp>(builder.getUnknownLoc(),
                                               aerFuncTable.at("aer_state"),
                                               ValueRange{}).getResult(0);
  
  patterns.add<FinalizeConversionPat,
               QubitDeclConversionPat,
               BuiltinUConversionPat,
               BuiltinCXConversionPat,
               MeasureConversionPat
               >(
    context, typeConverter, aerState);
  moduleOp->walk([](quir::MeasureOp op) {
      op.print(llvm::outs()); llvm::outs() << "\n";
  });
  moduleOp->walk([](quir::Builtin_UOp op) {
      op.print(llvm::outs()); llvm::outs() << "\n";
  });
  moduleOp->walk([](quir::BuiltinCXOp op) {
      op.print(llvm::outs()); llvm::outs() << "\n";
  });


  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    llvm::outs() << "Failed applyPartialConversion\n";
  }
} // QUIRToStdPass::runOnOperation()

llvm::StringRef QUIRToAERPass::getArgument() const {
  return "simulator-oq3-to-aer";
}

llvm::StringRef QUIRToAERPass::getDescription() const {
  return "Convert OQ3 ops to aer";
}

void QUIRToAERPass::declareAerFunctions(ModuleOp moduleOp) {
  using LLVM::LLVMFunctionType;

  aerFuncTable.clear();

  OpBuilder builder(moduleOp);

  auto registerFunc = [&](const char* name, LLVMFunctionType ty) {
    const auto loc = builder.getUnknownLoc();
    const auto f = builder.create<LLVM::LLVMFuncOp>(loc, name, ty);
    aerFuncTable.insert({name, f});
  };

  auto context = moduleOp->getContext();
  builder.setInsertionPointToStart(moduleOp.getBody());
  // common types
  const auto voidType = LLVM::LLVMVoidType::get(context);
  const auto i8Type = IntegerType::get(context, 8);
  const auto i64Type = IntegerType::get(context, 64);
  const auto aerStateType = LLVM::LLVMPointerType::get(i8Type);
  const auto strType = LLVM::LLVMPointerType::get(i8Type);
  // @aer_state(...) -> i8*
  const auto aerStateFunType = LLVMFunctionType::get(aerStateType, {}, true);
  registerFunc("aer_state", aerStateFunType);
  // @aer_state_configure(i8* noundef, i8* noundef, i8* noundef) -> void
  const auto aerStateConfigureType = LLVMFunctionType::get(voidType,
                                                           {strType, strType, strType});
  registerFunc("aer_state_configure", aerStateConfigureType);
  // @aer_allocate_qubits(i8* noundef, i64 noundef) -> i64
  const auto aerAllocQubitsType = LLVMFunctionType::get(i64Type,
                                                        {aerStateType, i64Type});
  registerFunc("aer_allocate_qubits", aerAllocQubitsType);
  // @aer_state_initialize(...) -> i8*
  const auto aerStateInitType = LLVMFunctionType::get(aerStateType, {}, true);
  registerFunc("aer_state_initialize", aerStateInitType);
  // @aer_state_finalize(i8* noundef) -> void
  const auto aerStateFinalizeType = LLVMFunctionType::get(voidType, aerStateType);
  registerFunc("aer_state_finalize", aerStateFinalizeType);
}

} // namespace qssc::targets::simulator::conversion

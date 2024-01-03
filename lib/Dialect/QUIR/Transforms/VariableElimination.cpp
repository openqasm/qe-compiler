//===- VariableElimination.cpp - Lower and eliminate variables --*- C++ -*-===//
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
///  This file implements the passes for converting QUIR variables to memref
///  operations and eliminating them where possible.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/VariableElimination.h"

#include "Conversion/OQ3ToStandard/OQ3ToStandard.h"
#include "Conversion/QUIRToStandard/VariablesToGlobalMemRefConversion.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <optional>
#include <utility>

namespace mlir {
void affineScalarReplaceCopy(mlir::func::FuncOp f, DominanceInfo &domInfo,
                             PostDominanceInfo &postDomInfo);
} // namespace mlir

namespace mlir::quir {

namespace {
std::optional<Type> convertCBitType(quir::CBitType t) {

  if (t.getWidth() <= 64)
    return IntegerType::get(t.getContext(), t.getWidth());

  return std::nullopt;
}

template <typename T>
std::optional<Type> legalizeType(T t) {
  return t;
}

class CBitTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

public:
  CBitTypeConverter() {
    addConversion(convertCBitType);
    addConversion(legalizeType<mlir::quir::AngleType>);
    addConversion(legalizeType<mlir::IntegerType>);
  }
};

struct MemrefGlobalToAllocaPattern
    : public OpRewritePattern<mlir::memref::GetGlobalOp> {
  MemrefGlobalToAllocaPattern(MLIRContext *context, mlir::Operation *toplevel)
      : OpRewritePattern<mlir::memref::GetGlobalOp>(context, /* benefit*/ 1),
        toplevel(toplevel) {}

  LogicalResult matchAndRewrite(mlir::memref::GetGlobalOp,
                                mlir::PatternRewriter &rewriter) const override;

  mlir::Operation *toplevel;
};

/// Materialize OQ3 casts to !quir.angle into a new cast when the argument can
/// be type-converted to integer.
struct MaterializeIntToAngleCastPattern
    : public OpConversionPattern<oq3::CastOp> {
  explicit MaterializeIntToAngleCastPattern(MLIRContext *ctx,
                                            mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(oq3::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!adaptor.getArg().getType().isIntOrIndexOrFloat() ||
        !castOp.getOut().getType().isa<mlir::quir::AngleType>())
      return failure();

    rewriter.replaceOpWithNewOp<oq3::CastOp>(castOp, castOp.getOut().getType(),
                                             adaptor.getArg());

    return success();
  } // matchAndRewrite
};  // struct MaterializeIntToAngleCastPattern

template <typename OperationType>
struct MaterializeBitOpForInt : public OpConversionPattern<OperationType> {
  explicit MaterializeBitOpForInt(MLIRContext *ctx,
                                  mlir::TypeConverter &typeConverter)
      : OpConversionPattern<OperationType>(typeConverter, ctx, /*benefit=*/1) {}
  LogicalResult
  matchAndRewrite(OperationType op, typename OperationType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    for (auto val : adaptor.getOperands())
      if (!val.getType().isSignlessInteger())
        return failure();
    if (!op.getResult().getType().isSignlessInteger())
      return failure();

    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });

    return success();
  }
};

mlir::LogicalResult convertQuirVariables(mlir::MLIRContext &context,
                                         mlir::Operation *top,
                                         bool externalizeOutputVariables) {

  // This conversion step gets rid of QUIR variables and classical bit
  // registers. These two concepts should be in the OpenQASM 3 dialect.
  // Effectively, this conversion is from (what should become) the OpenQASM 3
  // dialect to QUIR.
  // TODO transform into the OpenQASM 3 to QUIR lowering step.

  ConversionTarget target(context);
  CBitTypeConverter typeConverter;

  // Only convert QUIR variable operations
  target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect,
                         memref::MemRefDialect, scf::SCFDialect,
                         mlir::func::FuncDialect, affine::AffineDialect>();
  target.addIllegalOp<oq3::DeclareVariableOp, oq3::VariableAssignOp,
                      oq3::VariableLoadOp>();
  // TODO add additional QUIR variable operations here
  RewritePatternSet patterns(&context);

  quir::populateVariableToGlobalMemRefConversionPatterns(
      patterns, typeConverter, externalizeOutputVariables);

  // Convert `CBit` type and operations
  oq3::populateOQ3ToStandardConversionPatterns(typeConverter, patterns, false);
  // clang-format off
  target.addIllegalOp<
              oq3::CBitAssignBitOp,
              oq3::CBitNotOp,
              oq3::CBitRotLOp,
              oq3::CBitRotROp,
              oq3::CBitPopcountOp,
              oq3::CBitAndOp,
              oq3::CBitOrOp,
              oq3::CBitXorOp,
              oq3::CBitRShiftOp,
              oq3::CBitLShiftOp>();
  // clang-format on
  target.addDynamicallyLegalOp<oq3::CastOp>([](oq3::CastOp op) {
    if (op.getType().isa<mlir::quir::CBitType>() ||
        op.getArg().getType().isa<mlir::quir::CBitType>())
      return false;
    return true;
  });

  // Materialize CBitExtractBitOp and CBitInsertBitOp with integer operands.
  patterns.add<MaterializeBitOpForInt<oq3::CBitExtractBitOp>,
               MaterializeBitOpForInt<oq3::CBitInsertBitOp>>(&context,
                                                             typeConverter);

  target.addDynamicallyLegalOp<mlir::oq3::CBitExtractBitOp>(
      [](mlir::oq3::CBitExtractBitOp op) {
        if (op.getType().isa<mlir::quir::CBitType>() ||
            op.getOperand().getType().isa<mlir::quir::CBitType>())
          return false;

        return true;
      });
  target.addDynamicallyLegalOp<mlir::oq3::CBitInsertBitOp>(
      [](mlir::oq3::CBitInsertBitOp op) {
        if (op.getType().isa<mlir::quir::CBitType>() ||
            op.getOperand().getType().isa<mlir::quir::CBitType>())
          return false;

        return true;
      });

  // Support cbit to angle casts by materializing them into a new oq3.cast with
  // the argument type-converted to integer.
  patterns.add<MaterializeIntToAngleCastPattern>(&context, typeConverter);

  return applyPartialConversion(top, target, std::move(patterns));
}

LogicalResult MemrefGlobalToAllocaPattern::matchAndRewrite(
    mlir::memref::GetGlobalOp op, mlir::PatternRewriter &rewriter) const {

  // Check that the global memref is only used by this GetGlobalOp
  auto global =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
          op, op.getNameAttr());

  if (!global)
    return failure();
  if (!global.isPrivate())
    return failure();

  auto uses = global.getSymbolUses(toplevel);

  if (!uses)
    return failure();

  for (auto &use : uses.value()) {
    assert(use.getSymbolRef() == op.getNameAttr() && "found wrong symbol");
    if (use.getUser() != op) // other reference to the global memref
      return failure();
  }

  auto mrt = op.getResult().getType().dyn_cast<mlir::MemRefType>();

  assert(mrt && "expect result of a GetGlobalOp to be of MemRefType");
  if (!mrt)
    return failure();

  rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(
      op, mrt, global.getAlignmentAttr());
  rewriter.eraseOp(global);
  return success();
}

mlir::LogicalResult
convertIsolatedMemrefGlobalToAlloca(mlir::MLIRContext &context,
                                    mlir::Operation *top) {

  RewritePatternSet patterns(&context);

  patterns.add<MemrefGlobalToAllocaPattern>(&context, top);
  mlir::GreedyRewriteConfig config;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  config.useTopDownTraversal = true;

  return applyPatternsAndFoldGreedily(top, std::move(patterns), config);
}

struct RemoveAllocaWithIsolatedStoresPattern
    : public OpRewritePattern<mlir::memref::AllocaOp> {
  RemoveAllocaWithIsolatedStoresPattern(MLIRContext *context,
                                        mlir::Operation *toplevel)
      : OpRewritePattern<mlir::memref::AllocaOp>(context, /* benefit*/ 1),
        toplevel(toplevel) {}

  LogicalResult matchAndRewrite(mlir::memref::AllocaOp,
                                mlir::PatternRewriter &rewriter) const override;

  mlir::Operation *toplevel;
};

LogicalResult RemoveAllocaWithIsolatedStoresPattern::matchAndRewrite(
    mlir::memref::AllocaOp op, mlir::PatternRewriter &rewriter) const {

  llvm::SmallVector<mlir::Operation *> usersToErase;

  // Check that the only users are store operations
  // push to small vector for erasing
  for (auto *user : op.getResult().getUsers()) {
    usersToErase.push_back(user);
    if (!mlir::isa<mlir::affine::AffineStoreOp>(user))
      return failure();
  }

  // Drop all users
  for (auto *user : usersToErase)
    rewriter.eraseOp(user);

  // and remove the alloca
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult dropAllocaWithIsolatedStores(mlir::MLIRContext &context,
                                                 mlir::Operation *top) {

  RewritePatternSet patterns(&context);

  patterns.add<RemoveAllocaWithIsolatedStoresPattern>(&context, top);
  mlir::GreedyRewriteConfig config;

  config.useTopDownTraversal = true;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  return applyPatternsAndFoldGreedily(top, std::move(patterns), config);
}

} // anonymous namespace

void VariableEliminationPass::runOnOperation() {

  if (failed(convertQuirVariables(getContext(), getOperation(),
                                  externalizeOutputVariables)))
    return signalPassFailure();

  if (failed(convertIsolatedMemrefGlobalToAlloca(getContext(), getOperation())))
    return signalPassFailure();

  auto &domInfo = getAnalysis<DominanceInfo>();
  auto &postDomInfo = getAnalysis<PostDominanceInfo>();

  WalkResult const result = getOperation()->walk([&](mlir::func::FuncOp func) {
    mlir::affine::affineScalarReplace(func, domInfo, postDomInfo);

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();

  if (failed(dropAllocaWithIsolatedStores(getContext(), getOperation())))
    return signalPassFailure();
}

llvm::StringRef VariableEliminationPass::getArgument() const {
  return "quir-eliminate-variables";
}

llvm::StringRef VariableEliminationPass::getDescription() const {
  return "Replace QUIR variables by memref and simplify with store-forwarding.";
}

} // namespace mlir::quir

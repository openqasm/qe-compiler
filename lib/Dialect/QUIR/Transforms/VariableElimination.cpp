//===- VariableElimination.cpp - Lower and eliminate variables --*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "Conversion/QUIRToStandard/CBitOperations.h"
#include "Conversion/QUIRToStandard/QUIRCast.h"
#include "Conversion/QUIRToStandard/VariablesToGlobalMemRefConversion.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
void affineScalarReplaceCopy(FuncOp f, DominanceInfo &domInfo,
                             PostDominanceInfo &postDomInfo);
} // namespace mlir

namespace mlir::quir {

namespace {
Optional<Type> convertCbitType(quir::CBitType t) {

  if (t.getWidth() <= 64)
    return IntegerType::get(t.getContext(), t.getWidth());

  return llvm::None;
}

template <typename T>
Optional<Type> legalizeType(T t) {
  return t;
}

class CBitTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

public:
  CBitTypeConverter() {
    addConversion(convertCbitType);
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

/// Materialize quir casts to !quir.angle into a new cast when the argument can
/// be type-converted to integer.
struct MaterializeIntToAngleCastPattern : public OpConversionPattern<CastOp> {
  explicit MaterializeIntToAngleCastPattern(MLIRContext *ctx,
                                            mlir::TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, CastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!adaptor.arg().getType().isIntOrIndexOrFloat() ||
        !castOp.out().getType().isa<mlir::quir::AngleType>())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::quir::CastOp>(
        castOp, castOp.out().getType(), adaptor.arg());

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

} // namespace

static mlir::LogicalResult
convertQuirVariables(mlir::MLIRContext &context, mlir::Operation *top,
                     bool externalizeOutputVariables) {

  // This conversion step gets rid of QUIR variables and classical bit
  // registers. These two concepts should be in the OpenQASM 3 dialect.
  // Effectively, this conversion is from (what should become) the OpenQASM 3
  // dialect to QUIR.
  // TODO transform into the OpenQASM 3 to QUIR lowering step.

  ConversionTarget target(context);
  CBitTypeConverter typeConverter;

  // Only convert QUIR variable operations
  target.addLegalDialect<
      arith::ArithmeticDialect, LLVM::LLVMDialect, memref::MemRefDialect,
      scf::SCFDialect, StandardOpsDialect, quir::QUIRDialect, AffineDialect>();
  target.addIllegalOp<oq3::DeclareVariableOp>();
  target.addIllegalOp<oq3::AssignVariableOp>();
  target.addIllegalOp<quir::UseVariableOp>();
  // TODO add additional QUIR variable operations here
  RewritePatternSet patterns(&context);

  quir::populateVariableToGlobalMemRefConversionPatterns(
      patterns, typeConverter, externalizeOutputVariables);

  // Convert CBit type and operations
  quir::populateCBitOperationsPatterns(patterns, typeConverter, false);
  // TODO transform to making the OpenQASM dialect invalid
  target.addIllegalOp<quir::AssignCbitBitOp>();
  target.addIllegalOp<quir::Cbit_NotOp>();
  target.addIllegalOp<quir::Cbit_RotLOp>();
  target.addIllegalOp<quir::Cbit_RotROp>();
  target.addIllegalOp<quir::Cbit_PopcountOp>();
  target.addIllegalOp<quir::Cbit_AndOp>();
  target.addIllegalOp<quir::Cbit_OrOp>();
  target.addIllegalOp<quir::Cbit_XorOp>();
  target.addIllegalOp<quir::Cbit_RshiftOp>();
  target.addIllegalOp<quir::Cbit_LshiftOp>();

  // TODO move quir.cast and patterns for CBit types into OpenQASM 3 dialect.
  quir::populateQUIRCastPatterns(patterns, typeConverter);
  // QUIR Casts from / to cbit must be converted
  target.addDynamicallyLegalOp<mlir::quir::CastOp>([](mlir::quir::CastOp op) {
    if (op.getType().isa<mlir::quir::CBitType>() ||
        op.arg().getType().isa<mlir::quir::CBitType>())
      return false;
    return true;
  });

  // Materialize Cbit_ExtractBitOp and Cbit_InsertBitOp with integer operands.
  patterns.insert<MaterializeBitOpForInt<quir::Cbit_ExtractBitOp>>(
      &context, typeConverter);
  patterns.insert<MaterializeBitOpForInt<quir::Cbit_InsertBitOp>>(
      &context, typeConverter);

  target.addDynamicallyLegalOp<mlir::quir::Cbit_ExtractBitOp>(
      [](mlir::quir::Cbit_ExtractBitOp op) {
        if (op.getType().isa<mlir::quir::CBitType>() ||
            op.operand().getType().isa<mlir::quir::CBitType>())
          return false;

        return true;
      });
  target.addDynamicallyLegalOp<mlir::quir::Cbit_InsertBitOp>(
      [](mlir::quir::Cbit_InsertBitOp op) {
        if (op.getType().isa<mlir::quir::CBitType>() ||
            op.operand().getType().isa<mlir::quir::CBitType>())
          return false;

        return true;
      });

  // Support cbit to angle casts by materializing them into a new quir.cast with
  // the argument type-converted to integer.
  patterns.insert<MaterializeIntToAngleCastPattern>(&context, typeConverter);

  return applyPartialConversion(top, target, std::move(patterns));
}

namespace {
LogicalResult MemrefGlobalToAllocaPattern::matchAndRewrite(
    mlir::memref::GetGlobalOp op, mlir::PatternRewriter &rewriter) const {

  // Check that the global memref is only used by this GetGlobalOp
  auto global =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
          op, op.nameAttr());

  if (!global)
    return failure();
  if (!global.isPrivate())
    return failure();

  auto uses = global.getSymbolUses(toplevel);

  if (!uses)
    return failure();

  for (auto &use : uses.getValue()) {
    assert(use.getSymbolRef() == op.nameAttr() && "found wrong symbol");
    if (use.getUser() != op) // other reference to the global memref
      return failure();
  }

  auto mrt = op.result().getType().dyn_cast<mlir::MemRefType>();

  assert(mrt && "expect result of a GetGlobalOp to be of MemRefType");
  if (!mrt)
    return failure();

  rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, mrt,
                                                      global.alignmentAttr());
  rewriter.eraseOp(global);
  return success();
}
} // namespace

static mlir::LogicalResult
convertIsolatedMemrefGlobalToAlloca(mlir::MLIRContext &context,
                                    mlir::Operation *top) {

  RewritePatternSet patterns(&context);

  patterns.insert<MemrefGlobalToAllocaPattern>(&context, top);
  mlir::GreedyRewriteConfig config;

  config.useTopDownTraversal = true;

  return applyPatternsAndFoldGreedily(top, std::move(patterns), config);
}

namespace {
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

  // Check that the only users are store operations
  for (auto *user : op.getResult().getUsers())
    if (!mlir::isa<mlir::AffineStoreOp>(user))
      return failure();

  // Drop all users
  for (auto *user : op.getResult().getUsers())
    rewriter.eraseOp(user);

  // and remove the alloca
  rewriter.eraseOp(op);
  return success();
}
} // namespace

static mlir::LogicalResult
dropAllocaWithIsolatedStores(mlir::MLIRContext &context, mlir::Operation *top) {

  RewritePatternSet patterns(&context);

  patterns.insert<RemoveAllocaWithIsolatedStoresPattern>(&context, top);
  mlir::GreedyRewriteConfig config;

  config.useTopDownTraversal = true;

  return applyPatternsAndFoldGreedily(top, std::move(patterns), config);
}

void VariableEliminationPass::runOnOperation() {

  if (failed(convertQuirVariables(getContext(), getOperation(),
                                  externalizeOutputVariables)))
    return signalPassFailure();

  if (failed(convertIsolatedMemrefGlobalToAlloca(getContext(), getOperation())))
    return signalPassFailure();

  auto &domInfo = getAnalysis<DominanceInfo>();
  auto &postDomInfo = getAnalysis<PostDominanceInfo>();

  WalkResult result = getOperation()->walk([&](mlir::FuncOp func) {
    // TODO LLVM 15+: Use MLIR's builtin affineScalarReplace, which is fixed
    // there
    mlir::affineScalarReplaceCopy(func, domInfo, postDomInfo);

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

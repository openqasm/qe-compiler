//===- VariableElimination.cpp - Lower and eliminate variables --*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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

#include "Conversion/OQ3ToStandard/OQ3ToStandard.h"
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
Optional<Type> convertCBitType(quir::CBitType t) {

  if (t.getWidth() <= 64)
    return IntegerType::get(t.getContext(), t.getWidth());

  return llvm::None;
}

template <typename T> Optional<Type> legalizeType(T t) { return t; }

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

    if (!adaptor.arg().getType().isIntOrIndexOrFloat() ||
        !castOp.out().getType().isa<mlir::quir::AngleType>())
      return failure();

    rewriter.replaceOpWithNewOp<oq3::CastOp>(castOp, castOp.out().getType(),
                                             adaptor.arg());

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
  target.addLegalDialect<arith::ArithmeticDialect, LLVM::LLVMDialect,
                         memref::MemRefDialect, scf::SCFDialect,
                         StandardOpsDialect, AffineDialect>();
  target.addIllegalOp<oq3::DeclareVariableOp, oq3::VariableAssignOp,
                      oq3::UseVariableOp>();
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
        op.arg().getType().isa<mlir::quir::CBitType>())
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
            op.operand().getType().isa<mlir::quir::CBitType>())
          return false;

        return true;
      });
  target.addDynamicallyLegalOp<mlir::oq3::CBitInsertBitOp>(
      [](mlir::oq3::CBitInsertBitOp op) {
        if (op.getType().isa<mlir::quir::CBitType>() ||
            op.operand().getType().isa<mlir::quir::CBitType>())
          return false;

        return true;
      });

  // Support cbit to angle casts by materializing them into a new oq3.cast with
  // the argument type-converted to integer.
  patterns.add<MaterializeIntToAngleCastPattern>(&context, typeConverter);

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

  patterns.add<MemrefGlobalToAllocaPattern>(&context, top);
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

  patterns.add<RemoveAllocaWithIsolatedStoresPattern>(&context, top);
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

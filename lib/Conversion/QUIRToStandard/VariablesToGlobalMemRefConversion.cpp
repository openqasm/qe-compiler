//===- VariablesToGobalMemRefConversion.cpp ---------------------*- C++ -*-===//
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
/// \file
/// This file implements patterns for lowering QUIR variable declarations,
/// variable use, and variable assignments to std. For that purpose, it
/// introduces a global variable for each QUIR variable declaration via a
/// GlobalMemRef. All variable references and assignments are converted into
/// load and store op against the global memrefs.
///
//===----------------------------------------------------------------------===//

#include "Conversion/QUIRToStandard/VariablesToGlobalMemRefConversion.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::oq3;
using namespace mlir::quir;

namespace {
std::optional<mlir::memref::GlobalOp>
createGlobalMemrefOp(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Operation *insertionAnchor, mlir::MemRefType type,
                     llvm::StringRef name) {

  // place memref::GlobalOps at the top of the surrounding module.
  // (note: required by llvm.mlir.global, which this is lowered to)
  mlir::OpBuilder::InsertionGuard g(builder);
  auto containingModule = insertionAnchor->getParentOfType<mlir::ModuleOp>();

  if (!containingModule) {
    insertionAnchor->emitOpError("Missing a global ModuleOp container.");
    return std::nullopt;
  }

  builder.setInsertionPoint(&containingModule.front());

  // "private" symbols are visible only to the closest symbol table.
  // thus, any references to the symbol can be observed in this module, and
  // LLVM's optimization passes have free reign (e.g., to completely remove
  // any load/store of the variable).
  auto const symbolVisibility = builder.getStringAttr("private");

  return builder.create<mlir::memref::GlobalOp>(
      loc, name, symbolVisibility, type,
      /* no initialization */ builder.getUnitAttr(), /* constant= */ false,
      /* alignment= */ nullptr);
}

struct VariableDeclarationConversionPattern
    : public OpConversionPattern<DeclareVariableOp> {
  explicit VariableDeclarationConversionPattern(MLIRContext *ctx,
                                                TypeConverter &typeConverter,
                                                bool externalizeOutputVariables)
      : OpConversionPattern<DeclareVariableOp>(typeConverter, ctx,
                                               /*benefit=*/1),
        externalizeOutputVariables(externalizeOutputVariables) {}

  bool const externalizeOutputVariables;

  LogicalResult
  matchAndRewrite(DeclareVariableOp declareOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto declarationType = declareOp.type();
    auto convertedType = typeConverter->convertType(declareOp.type());
    if (convertedType)
      declarationType = convertedType;

    // a scalar is a MemRef of dimensionality 0 (shape is an empty list)
    auto const memRefType =
        mlir::MemRefType::get(llvm::ArrayRef<int64_t>(), declarationType);
    assert(memRefType && "failed to instantiate a MemRefType, likely trying "
                         "with invalid element type");

    auto gmoOrNone =
        createGlobalMemrefOp(rewriter, declareOp.getLoc(), declareOp,
                             memRefType, declareOp.getName());
    if (!gmoOrNone)
      return failure();

    auto gmo = gmoOrNone.getValue();

    if (externalizeOutputVariables && declareOp.isOutputVariable()) {
      // for generating defined symbols, global memrefs need an initializer
      auto rankedTensorType =
          mlir::RankedTensorType::get(llvm::ArrayRef<int64_t>{}, convertedType);
      auto elementInitializerAttr = rewriter.getIntegerAttr(convertedType, 0);
      auto initializerAttr = mlir::DenseElementsAttr::get(
          rankedTensorType,
          llvm::ArrayRef<mlir::Attribute>{elementInitializerAttr});

      gmo.initial_valueAttr(initializerAttr);
      gmo.setPublic();
    }

    rewriter.replaceOp(declareOp, mlir::ValueRange{});
    return success();
  }
};

struct ArrayDeclarationConversionPattern
    : public OpConversionPattern<DeclareArrayOp> {
  explicit ArrayDeclarationConversionPattern(MLIRContext *ctx,
                                             TypeConverter &typeConverter)
      : OpConversionPattern<DeclareArrayOp>(typeConverter, ctx,
                                            /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(DeclareArrayOp declareOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto declarationType = declareOp.type();
    auto convertedType = typeConverter->convertType(declareOp.type());
    if (convertedType)
      declarationType = convertedType;

    uint64_t num_elements = declareOp.num_elements().getZExtValue();
    auto const memRefType = mlir::MemRefType::get(
        llvm::ArrayRef<int64_t>{(int64_t)num_elements}, declareOp.type());
    assert(memRefType && "failed to instantiate a MemRefType, likely trying "
                         "with invalid element type");

    if (!createGlobalMemrefOp(rewriter, declareOp.getLoc(), declareOp,
                              memRefType, declareOp.getName()))
      return failure();
    rewriter.eraseOp(declareOp);

    return success();
  }
};

/// @brief Find an existing GetGlobalMemrefOp for a QUIR variable in the
/// surrounding function or create a new one at the head of the surrounding
/// function. The QUIR variable's declaration must already have been converted
/// into a GlobalMemrefOp.
/// @tparam QUIRVariableOp template parameter for the type of QUIRVariableOp
/// @param variableOp the variable operation to find or create a
/// GetGlobalMemrefOp for
/// @return a GetGlobalMemrefOp for the given variable op
template <class QUIRVariableOp>
std::optional<mlir::memref::GetGlobalOp>
findOrCreateGetGlobalMemref(QUIRVariableOp variableOp,
                            ConversionPatternRewriter &builder) {
  mlir::OpBuilder::InsertionGuard g(builder);

  auto globalMemrefOp =
      SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
          variableOp, variableOp.variable_nameAttr());

  if (!globalMemrefOp) {
    variableOp.emitOpError("Cannot lookup a variable declaration for " +
                           variableOp.variable_name());
    return std::nullopt;
  }

  auto surroundingFunction =
      variableOp->template getParentOfType<mlir::func::FuncOp>();
  if (!surroundingFunction) {
    variableOp.emitOpError("Variable use of " + variableOp.variable_name() +
                           " outside functions not supported");
    return std::nullopt;
  }

  // Search for an existing memref::GetGlobalOp in the surrounding function by
  // walking all the GlobalMemref's symbol uses.
  if (auto rangeOrNone = mlir::SymbolTable::getSymbolUses(
          /* symbol */ globalMemrefOp, /* inside */ surroundingFunction))
    for (auto &use : rangeOrNone.value())
      if (llvm::isa<mlir::memref::GetGlobalOp>(use.getUser()))
        return llvm::cast<mlir::memref::GetGlobalOp>(use.getUser());

  // Create new one at the top of the start of the function
  builder.setInsertionPointToStart(&surroundingFunction.getBody().front());

  return builder.create<mlir::memref::GetGlobalOp>(
      variableOp.getLoc(), globalMemrefOp.type(), globalMemrefOp.sym_name());
}

struct VariableUseConversionPattern
    : public OpConversionPattern<VariableLoadOp> {
  explicit VariableUseConversionPattern(MLIRContext *ctx,
                                        TypeConverter &typeConverter)
      : OpConversionPattern<VariableLoadOp>(typeConverter, ctx, /*benefit=*/1) {

  }

  LogicalResult
  matchAndRewrite(VariableLoadOp useOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto varRefOrNone = findOrCreateGetGlobalMemref(useOp, rewriter);
    if (!varRefOrNone)
      return failure();

    auto varRef = varRefOrNone.getValue();
    auto loadOp =
        rewriter.create<mlir::AffineLoadOp>(useOp.getLoc(), varRef.getResult());

    rewriter.replaceOp(useOp, {loadOp});
    return success();
  }
};

struct ArrayElementUseConversionPattern
    : public OpConversionPattern<UseArrayElementOp> {
  explicit ArrayElementUseConversionPattern(MLIRContext *ctx,
                                            TypeConverter &typeConverter)
      : OpConversionPattern<UseArrayElementOp>(typeConverter, ctx,
                                               /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(UseArrayElementOp useOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto varRefOrNone = findOrCreateGetGlobalMemref(useOp, rewriter);
    if (!varRefOrNone)
      return failure();
    auto varRef = varRefOrNone.getValue();

    auto indexOp = rewriter.create<mlir::arith::ConstantOp>(
        useOp.getLoc(), rewriter.getIndexType(), useOp.indexAttr());
    auto loadOp = rewriter.create<mlir::memref::LoadOp>(
        useOp.getLoc(), varRef.getResult(), mlir::ValueRange{indexOp});

    rewriter.replaceOp(useOp, {loadOp});
    return success();
  }
};

struct VariableAssignConversionPattern
    : public OpConversionPattern<VariableAssignOp> {
  explicit VariableAssignConversionPattern(MLIRContext *ctx,
                                           TypeConverter &typeConverter)
      : OpConversionPattern<VariableAssignOp>(typeConverter, ctx,
                                              /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(VariableAssignOp assignOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto varRefOrNone = findOrCreateGetGlobalMemref(assignOp, rewriter);
    if (!varRefOrNone)
      return failure();
    auto varRef = varRefOrNone.getValue();

    rewriter.create<mlir::AffineStoreOp>(
        assignOp.getLoc(), adaptor.assigned_value(), varRef.getResult(),
        mlir::ValueRange{});

    rewriter.eraseOp(assignOp);
    return success();
  }
};

struct ArrayElementAssignConversionPattern
    : public OpConversionPattern<AssignArrayElementOp> {
  explicit ArrayElementAssignConversionPattern(MLIRContext *ctx,
                                               TypeConverter &typeConverter)
      : OpConversionPattern<AssignArrayElementOp>(typeConverter, ctx,
                                                  /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(AssignArrayElementOp assignOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto varRefOrNone = findOrCreateGetGlobalMemref(assignOp, rewriter);

    if (!varRefOrNone)
      return failure();
    auto varRef = varRefOrNone.getValue();

    auto indexOp = rewriter.create<mlir::arith::ConstantOp>(
        assignOp.getLoc(), rewriter.getIndexType(), assignOp.indexAttr());

    rewriter.create<mlir::memref::StoreOp>(
        assignOp.getLoc(), adaptor.assigned_value(), varRef.getResult(),
        mlir::ValueRange{indexOp});

    rewriter.replaceOp(assignOp, mlir::ValueRange{});
    return success();
  }
};
} // anonymous namespace

void mlir::quir::populateVariableToGlobalMemRefConversionPatterns(
    RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    bool externalizeOutputVariables) {
  auto *ctx = patterns.getContext();
  assert(ctx);

  patterns.add<VariableDeclarationConversionPattern>(
      ctx, typeConverter, externalizeOutputVariables);
  // clang-format off
  patterns.add<
      VariableAssignConversionPattern,
      VariableUseConversionPattern,
      ArrayDeclarationConversionPattern,
      ArrayElementUseConversionPattern,
      ArrayElementAssignConversionPattern>(ctx, typeConverter);
  // clang-format on
}

//===- AngleConversion.cpp - Convert CallGateOp Angles --------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the pass for converting angles in CallGateOp
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/AngleConversion.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Transforms/Analysis.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quir;

namespace {

struct AngleConversion : public OpRewritePattern<quir::CallGateOp> {
  explicit AngleConversion(MLIRContext *ctx)
      : OpRewritePattern<quir::CallGateOp>(ctx) {}
  LogicalResult matchAndRewrite(quir::CallGateOp callGateOp,
                                PatternRewriter &rewriter) const override {
    // find the corresponding FuncOp
    Operation *findOp = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
        callGateOp, callGateOp.calleeAttr());
    if (!findOp)
      return failure();
    FuncOp funcOp = dyn_cast<FuncOp>(findOp);
    FunctionType fType = funcOp.getType();

    for (auto &pair : llvm::enumerate(callGateOp.getArgOperands())) {
      auto value = pair.value();
      auto index = pair.index();
      if (auto declOp = value.getDefiningOp<quir::ConstantOp>()) {
        // compare the angle type in FuncOp and callGateOp
        // and change the angle type in callGateOp if the types are different
        Type funcType = fType.getInput(index);
        if (value.getType() != funcType) {
          APFloat constVal = declOp.getAngleValueFromConstant();
          declOp.valueAttr(
              AngleAttr::get(callGateOp.getContext(), funcType, constVal));
          value.setType(funcType);
        }
      }
    }
    return success();
  }
}; // struct AngleConversion

} // end anonymous namespace

// Entry point for the pass.
void QUIRAngleConversionPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  patterns.add<AngleConversion>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    ; // TODO why would this call to applyPatternsAndFoldGreedily fail?
      // signalPassFailure();
  }
}

llvm::StringRef QUIRAngleConversionPass::getArgument() const {
  return "convert-quir-angles";
}
llvm::StringRef QUIRAngleConversionPass::getDescription() const {
  return "Convert the angle types in CallGateOp "
         "based on the corresponding FuncOp args";
}

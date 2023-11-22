//===- AngleConversion.cpp - Convert CallGateOp Angles --------*- C++ -*-===//
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
///  This file implements the pass for converting angles in CallGateOp
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/AngleConversion.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_map>

using namespace mlir;
using namespace mlir::quir;

namespace {

struct AngleConversion : public OpRewritePattern<quir::CallGateOp> {
  explicit AngleConversion(MLIRContext *ctx,
                           std::unordered_map<std::string, mlir::func::FuncOp> &functionOps)
      : OpRewritePattern<quir::CallGateOp>(ctx), functionOps_(functionOps) {}
  LogicalResult matchAndRewrite(quir::CallGateOp callGateOp,
                                PatternRewriter &rewriter) const override {
    // find the corresponding mlir::func::FuncOp
    auto findOp = functionOps_.find(callGateOp.getCalleeAttr().getValue().str());
    if (findOp == functionOps_.end())
      return failure();

    mlir::func::FuncOp funcOp = findOp->second;
    FunctionType fType = funcOp.getFunctionType();

    for (auto &pair : llvm::enumerate(callGateOp.getArgOperands())) {
      auto value = pair.value();
      auto index = pair.index();
      if (auto declOp = value.getDefiningOp<quir::ConstantOp>()) {
        // compare the angle type in mlir::func::FuncOp and callGateOp
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

private:
  std::unordered_map<std::string, mlir::func::FuncOp> &functionOps_;
}; // struct AngleConversion

} // end anonymous namespace

// Entry point for the pass.
void QUIRAngleConversionPass::runOnOperation() {

  auto *op = getOperation();
  op->walk(
      [&](mlir::func::FuncOp funcOp) { functionOps[funcOp.getSymName().str()] = funcOp; });

  RewritePatternSet patterns(&getContext());
  patterns.add<AngleConversion>(&getContext(), functionOps);

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
         "based on the corresponding mlir::func::FuncOp args";
}

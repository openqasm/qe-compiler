//===- AngleConversion.cpp - Convert CallGateOp Angles --------*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
  patterns.insert<AngleConversion>(&getContext());

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

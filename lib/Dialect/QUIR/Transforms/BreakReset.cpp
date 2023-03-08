//===- BreakReset.cpp - Break apart reset ops -------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
///  This file implements the pass for breaking apart reset ops into
///  control flow.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/BreakReset.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <unordered_set>

using namespace mlir;
using namespace mlir::quir;

namespace {
// This pattern merges qubit reset operations that can be parallelized into one
// operation.
struct BreakResetsPattern : public OpRewritePattern<ResetQubitOp> {

  explicit BreakResetsPattern(MLIRContext *ctx, uint numIterations,
                              uint delayCycles)
      : OpRewritePattern<ResetQubitOp>(ctx), numIterations_(numIterations),
        delayCycles_(delayCycles) {}

  LogicalResult matchAndRewrite(ResetQubitOp resetOp,
                                PatternRewriter &rewriter) const override {
    DeclareDurationOp durationOp;

    if (numIterations_ > 1 && delayCycles_ > 0) {
      std::string durationString = std::to_string(delayCycles_) + "dt";
      durationOp = rewriter.create<DeclareDurationOp>(
          resetOp.getLoc(), rewriter.getType<DurationType>(),
          StringRef(durationString));
    }

    // result of measurement in each iteration is number of qubits * i1
    std::vector<mlir::Type> typeVec(resetOp.qubits().size(),
                                    rewriter.getI1Type());

    for (uint iteration = 0; iteration < numIterations_; iteration++) {
      if (delayCycles_ > 0 && iteration > 0)
        for (auto qubit : resetOp.qubits())
          rewriter.create<DelayOp>(resetOp.getLoc(), durationOp.out(), qubit);

      auto measureOp = rewriter.create<MeasureOp>(
          resetOp.getLoc(), TypeRange(typeVec), resetOp.qubits());
      measureOp->setAttr(getNoReportRuntimeAttrName(), rewriter.getUnitAttr());

      size_t i = 0;
      for (auto qubit : resetOp.qubits()) {
        auto ifOp = rewriter.create<scf::IfOp>(resetOp.getLoc(),
                                               measureOp.getResult(i), false);
        auto savedInsertionPoint = rewriter.saveInsertionPoint();
        auto *thenBlock = ifOp.getBody(0);

        rewriter.setInsertionPointToStart(thenBlock);
        rewriter.create<CallGateOp>(resetOp.getLoc(), StringRef("x"),
                                    TypeRange{}, ValueRange{qubit});
        i++;
        rewriter.restoreInsertionPoint(savedInsertionPoint);
      }
    }

    rewriter.eraseOp(resetOp);
    return success();
  }

private:
  uint numIterations_;
  uint delayCycles_;
}; // BreakResetsPattern
} // anonymous namespace

void BreakResetPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  mlir::GreedyRewriteConfig config;

  // use cheaper top-down traversal (in this case, bottom-up would not behave
  // any differently)
  config.useTopDownTraversal = true;

  patterns.insert<BreakResetsPattern>(&getContext(), numIterations,
                                      delayCycles);

  if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config)))
    signalPassFailure();
} // BreakResetPass::runOnOperation

llvm::StringRef BreakResetPass::getArgument() const { return "break-reset"; }
llvm::StringRef BreakResetPass::getDescription() const {
  return "Break reset ops into repeated measure and conditional x gate calls";
}

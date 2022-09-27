//===- MergeMeasures.cpp - Merge measurement ops ----------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the pass for merging measurements into a single
///  measure op
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/MergeMeasures.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <unordered_set>

using namespace mlir;
using namespace mlir::quir;

namespace {
// This pattern matches on two MeasureOps that are only interspersed by
// classical non-control flow ops and merges them into one measure op
struct MeasureAndMeasurePattern : public OpRewritePattern<MeasureOp> {
  explicit MeasureAndMeasurePattern(MLIRContext *ctx)
      : OpRewritePattern<MeasureOp>(ctx) {}

  LogicalResult matchAndRewrite(MeasureOp measureOp,
                                PatternRewriter &rewriter) const override {
    llvm::Optional<Operation *> nextQuantumOp = nextQuantumOpOrNull(measureOp);
    if (!nextQuantumOp)
      return failure();

    auto nextMeasureOp = dyn_cast<MeasureOp>(*nextQuantumOp);
    if (!nextMeasureOp)
      return failure();

    // found a measure and a measure, now make sure they aren't working on the
    // same qubit and that we can resolve them both
    std::unordered_set<uint> measureIds;
    for (auto qubit : measureOp.qubits()) {
      llvm::Optional<uint> id = lookupQubitId(qubit);
      if (!id)
        return failure();
      measureIds.emplace(*id);
    }
    for (auto qubit : nextMeasureOp.qubits()) {
      llvm::Optional<uint> id = lookupQubitId(qubit);
      if (!id || measureIds.count(*id))
        return failure();
    }

    // good to merge
    std::vector<Type> typeVec;
    std::vector<Value> valVec;
    typeVec.reserve(measureOp.getNumResults() + nextMeasureOp.getNumResults());
    valVec.reserve(measureOp.getNumResults() + nextMeasureOp.getNumResults());

    typeVec.insert(typeVec.end(), measureOp.result_type_begin(),
                   measureOp.result_type_end());
    typeVec.insert(typeVec.end(), nextMeasureOp.result_type_begin(),
                   nextMeasureOp.result_type_end());
    valVec.insert(valVec.end(), measureOp.qubits().begin(),
                  measureOp.qubits().end());
    valVec.insert(valVec.end(), nextMeasureOp.qubits().begin(),
                  nextMeasureOp.qubits().end());

    auto mergedOp = rewriter.create<MeasureOp>(
        measureOp.getLoc(), TypeRange(typeVec), ValueRange(valVec));

    // dice the output so we can specify which results to replace
    auto iterSep = mergedOp.outs().begin() + measureOp.getNumResults();
    rewriter.replaceOp(measureOp,
                       ResultRange(mergedOp.outs().begin(), iterSep));
    rewriter.replaceOp(nextMeasureOp,
                       ResultRange(iterSep, mergedOp.outs().end()));
    return success();
  } // matchAndRewrite
};  // struct MeasureAndMeasurePattern
} // end anonymous namespace

void MergeMeasuresPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.insert<MeasureAndMeasurePattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();
} // runOnOperation

llvm::StringRef MergeMeasuresPass::getArgument() const {
  return "merge-measures";
}
llvm::StringRef MergeMeasuresPass::getDescription() const {
  return "Merge qubit-parallel measurement operations into a "
         "single measurement operation";
}

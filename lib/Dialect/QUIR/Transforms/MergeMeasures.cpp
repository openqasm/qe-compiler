//===- MergeMeasures.cpp - Merge measurement ops ----------------*- C++ -*-===//
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

#include <algorithm>
#include <unordered_set>

using namespace mlir;
using namespace mlir::quir;

namespace {
// This pattern matches on two MeasureOps that are only interspersed by
// classical non-control flow ops and merges them into one measure op

static void mergeMeasurements(PatternRewriter &rewriter, MeasureOp measureOp,
                              MeasureOp nextMeasureOp) {
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
  rewriter.replaceOp(measureOp, ResultRange(mergedOp.outs().begin(), iterSep));
  rewriter.replaceOp(nextMeasureOp,
                     ResultRange(iterSep, mergedOp.outs().end()));
}

struct MeasureAndMeasureLexographicalPattern
    : public OpRewritePattern<MeasureOp> {
  explicit MeasureAndMeasureLexographicalPattern(MLIRContext *ctx)
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
    mergeMeasurements(rewriter, measureOp, nextMeasureOp);

    return success();
  } // matchAndRewrite
};  // struct MeasureAndMeasureLexographicalPattern
} // end anonymous namespace

void MergeMeasuresLexographicalPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.insert<MeasureAndMeasureLexographicalPattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();
} // runOnOperation

llvm::StringRef MergeMeasuresLexographicalPass::getArgument() const {
  return "merge-measures-lexographical";
}
llvm::StringRef MergeMeasuresLexographicalPass::getDescription() const {
  return "Merge qubit-parallel measurement operations into a "
         "single measurement operation with lexographicalal ordering";
}

namespace {
// This pattern matches on two MeasureOps that are only interspersed by
// classical non-control flow ops and merges them into one measure op
struct MeasureAndMeasureTopologicalPattern
    : public OpRewritePattern<MeasureOp> {
  explicit MeasureAndMeasureTopologicalPattern(MLIRContext *ctx)
      : OpRewritePattern<MeasureOp>(ctx) {}

  LogicalResult matchAndRewrite(MeasureOp measureOp,
                                PatternRewriter &rewriter) const override {
    // Accumulate qubits in measurement set
    std::set<uint> currMeasureQubits = measureOp.getOperatedQubits();

    // Find the next measurement operation accumulating qubits along the
    // topological path if it exists
    auto [nextMeasureOpt, observedQubits] =
        QubitOpInterface::getNextQubitOpOfTypeWithQubits<MeasureOp>(measureOp);
    if (!nextMeasureOpt.hasValue())
      return failure();

    // If any qubit along path touches the same qubits we cannot merge the next
    // measurement.
    currMeasureQubits.insert(observedQubits.begin(), observedQubits.end());

    // found a measure and a measure, now make sure they aren't working on the
    // same qubit and that we can resolve them both
    MeasureOp nextMeasureOp = nextMeasureOpt.getValue();
    auto nextMeasureQubits = nextMeasureOp.getOperatedQubits();

    // If there is an intersection we cannot merge
    std::set<int> mergeMeasureIntersection;
    std::set_intersection(currMeasureQubits.begin(), currMeasureQubits.end(),
                          nextMeasureQubits.begin(), nextMeasureQubits.end(),
                          std::inserter(mergeMeasureIntersection,
                                        mergeMeasureIntersection.begin()));

    if (!mergeMeasureIntersection.empty())
      return failure();

    // good to merge
    mergeMeasurements(rewriter, measureOp, nextMeasureOp);

    return success();
  } // matchAndRewrite
};  // struct MeasureAndMeasureTopologicalPattern
} // end anonymous namespace

void MergeMeasuresTopologicalPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.insert<MeasureAndMeasureTopologicalPattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();
} // runOnOperation

llvm::StringRef MergeMeasuresTopologicalPass::getArgument() const {
  return "merge-measures-topological";
}
llvm::StringRef MergeMeasuresTopologicalPass::getDescription() const {
  return "Merge qubit-parallel measurement operations into a "
         "single measurement operation with topological ordering";
}

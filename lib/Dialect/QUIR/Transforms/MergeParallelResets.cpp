//===- MergeParallelResets.cpp - Merge reset ops ----------------*- C++ -*-===//
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
///  This file implements the pass for merging resets into a single
///  reset op
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/MergeParallelResets.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <unordered_set>

using namespace mlir;
using namespace mlir::quir;

namespace {

// This pattern merges qubit reset operations that can be parallelized into a
// single reset op lexicographically.
struct MergeResetsLexicographicPattern : public OpRewritePattern<ResetQubitOp> {

  explicit MergeResetsLexicographicPattern(MLIRContext *ctx)
      : OpRewritePattern<ResetQubitOp>(ctx) {}

  LogicalResult matchAndRewrite(ResetQubitOp resetOp,
                                PatternRewriter &rewriter) const override {
    std::unordered_set<uint> qubitIds;
    qubitIds.reserve(resetOp.qubits().size());

    for (auto qubit : resetOp.qubits()) {
      auto id = lookupQubitId(qubit);
      if (!id)
        return failure();
      qubitIds.emplace(*id);
    }

    // identify additional reset operations that may happen in parallel and can
    // be merged
    auto resetQubitOperands = resetOp.qubitsMutable();

    llvm::Optional<Operation *> nextQuantumOp = nextQuantumOpOrNull(resetOp);
    if (!nextQuantumOp)
      return failure();

    auto nextResetOp = dyn_cast<ResetQubitOp>(*nextQuantumOp);
    if (!nextResetOp)
      return failure();

    // check if we can add this reset
    if (!std::all_of(nextResetOp.qubits().begin(), nextResetOp.qubits().end(),
                     [&](auto qubit) {
                       // can merge this adjacent qubit reset op when we can
                       // lookup all of its qubits' ids and these are not
                       // overlapping with the qubit ids in resetOp or other
                       // qubit resets that we want to merge
                       auto id = lookupQubitId(qubit);
                       return id && (qubitIds.count(*id) == 0);
                     }))
      return failure();

    // good to merge
    for (auto qubit : nextResetOp.qubits())
      resetQubitOperands.append(qubit);
    rewriter.eraseOp(nextResetOp);
    return success();
  }

}; // MergeResetsLexicographicPattern
} // anonymous namespace

void MergeResetsLexicographicPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  mlir::GreedyRewriteConfig config;

  // use cheaper top-down traversal (in this case, bottom-up would not behave
  // any differently)
  config.useTopDownTraversal = true;

  patterns.insert<MergeResetsLexicographicPattern>(&getContext());

  if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config)))
    signalPassFailure();
} // MergeResetsLexicographicPass::runOnOperation

llvm::StringRef MergeResetsLexicographicPass::getArgument() const {
  return "merge-resets-lexicographic";
}
llvm::StringRef MergeResetsLexicographicPass::getDescription() const {
  return "Merge qubit reset ops that can be parallelized into a "
         "single operation lexicographically.";
}

namespace {
// This pattern merges qubit reset operations that can be parallelized into a
// single reset op topologically.
struct MergeResetsTopologicalPattern : public OpRewritePattern<ResetQubitOp> {

  explicit MergeResetsTopologicalPattern(MLIRContext *ctx)
      : OpRewritePattern<ResetQubitOp>(ctx) {}

  LogicalResult matchAndRewrite(ResetQubitOp resetOp,
                                PatternRewriter &rewriter) const override {
    // Accumulate qubits in reset set
    std::set<uint> curQubits = resetOp.getOperatedQubits();

    // Find the next measurement operation accumulating qubits along the
    // topological path if it exists
    auto [nextResetOpt, observedQubits] =
        QubitOpInterface::getNextQubitOpOfTypeWithQubits<ResetQubitOp>(resetOp);
    if (!nextResetOpt.hasValue())
      return failure();

    // If any qubit along path touches the same qubits we cannot merge the next
    // reset.
    curQubits.insert(observedQubits.begin(), observedQubits.end());

    auto resetQubitOperands = resetOp.qubitsMutable();

    // found a measure and a measure, now make sure they aren't working on the
    // same qubit and that we can resolve them both
    ResetQubitOp nextResetOp = nextResetOpt.getValue();
    auto nextQubits = nextResetOp.getOperatedQubits();

    // If there is an intersection we cannot merge
    std::set<int> mergeIntersection;
    std::set_intersection(
        curQubits.begin(), curQubits.end(), nextQubits.begin(),
        nextQubits.end(),
        std::inserter(mergeIntersection, mergeIntersection.begin()));

    if (!mergeIntersection.empty())
      return failure();

    // good to merge
    for (auto qubit : nextResetOp.qubits())
      resetQubitOperands.append(qubit);
    rewriter.eraseOp(nextResetOp);
    return success();
  }

}; // MergeResetsTopologicalPattern
} // anonymous namespace

void MergeResetsTopologicalPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  mlir::GreedyRewriteConfig config;

  // use cheaper top-down traversal (in this case, bottom-up would not behave
  // any differently)
  config.useTopDownTraversal = true;

  patterns.insert<MergeResetsTopologicalPattern>(&getContext());

  if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config)))
    signalPassFailure();
} // MergeResetsTopologicalPass::runOnOperation

llvm::StringRef MergeResetsTopologicalPass::getArgument() const {
  return "merge-resets-topological";
}
llvm::StringRef MergeResetsTopologicalPass::getDescription() const {
  return "Merge qubit reset ops that can be parallelized into a "
         "single operation topologically.";
}

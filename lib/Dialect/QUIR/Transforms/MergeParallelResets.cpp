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
    qubitIds.reserve(resetOp.getQubits().size());

    for (auto qubit : resetOp.getQubits()) {
      auto id = lookupQubitId(qubit);
      if (!id)
        return failure();
      qubitIds.emplace(*id);
    }

    // identify additional reset operations that may happen in parallel and can
    // be merged
    auto resetQubitOperands = resetOp.getQubitsMutable();

    std::optional<Operation *> nextQuantumOp = nextQuantumOpOrNull(resetOp);
    if (!nextQuantumOp)
      return failure();

    auto nextResetOp = dyn_cast<ResetQubitOp>(*nextQuantumOp);
    if (!nextResetOp)
      return failure();

    // check if we can add this reset
    if (!std::all_of(nextResetOp.getQubits().begin(),
                     nextResetOp.getQubits().end(), [&](auto qubit) {
                       // can merge this adjacent qubit reset op when we can
                       // lookup all of its qubits' ids and these are not
                       // overlapping with the qubit ids in resetOp or other
                       // qubit resets that we want to merge
                       auto id = lookupQubitId(qubit);
                       return id && (qubitIds.count(*id) == 0);
                     }))
      return failure();

    // good to merge
    for (auto qubit : nextResetOp.getQubits())
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
  // Disable to improve performance
  config.enableRegionSimplification = false;

  patterns.add<MergeResetsLexicographicPattern>(&getContext());

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
    // Find the next measurement operation accumulating qubits along the
    // topological path if it exists
    auto [nextResetOpt, observedQubits] =
        QubitOpInterface::getNextQubitOpOfTypeWithQubits<ResetQubitOp>(resetOp);
    if (!nextResetOpt.has_value())
      return failure();

    ResetQubitOp nextResetOp = nextResetOpt.value();

    // There are 2 possible merge directions; we can hoist the next reset
    // to merge with this one (if nothing uses the future qubit between here
    // and the next reset), or we can delay this reset until the next one (if
    // nothing uses this qubit between here and the next reset). Both options
    // are possible because the reset operation doesn't produce a result, so
    // deferring it is trivial.

    // Get the qubits used in both this reset and the next
    auto curQubits = resetOp.getOperatedQubits();
    auto nextQubits = nextResetOp.getOperatedQubits();

    // Form a forward set (all qubits used between here and the next
    // reset, including this operation) and a backward set (all qubits
    // used between here and the next reset, including the next)
    std::set<uint> fwdQubits = curQubits;
    fwdQubits.insert(observedQubits.begin(), observedQubits.end());
    std::set<uint> backQubits = nextQubits;
    backQubits.insert(observedQubits.begin(), observedQubits.end());

    // If any qubit along path touches the same qubits we cannot merge in
    // that direction, so check for intersections
    std::set<int> mergeFwdIntersection;
    std::set_intersection(
        fwdQubits.begin(), fwdQubits.end(), nextQubits.begin(),
        nextQubits.end(),
        std::inserter(mergeFwdIntersection, mergeFwdIntersection.begin()));

    std::set<int> mergeBackIntersection;
    std::set_intersection(
        backQubits.begin(), backQubits.end(), curQubits.begin(),
        curQubits.end(),
        std::inserter(mergeBackIntersection, mergeBackIntersection.begin()));

    if (!mergeFwdIntersection.empty() && !mergeBackIntersection.empty())
      // Can't merge in EITHER direction
      return failure();

    // good to merge one way or the other. Prefer hoisting the next reset.
    if (mergeFwdIntersection.empty()) {
      // Hoist the next reset into this one
      auto resetQubitOperands = resetOp.getQubitsMutable();
      for (auto qubit : nextResetOp.getQubits())
        resetQubitOperands.append(qubit);
      rewriter.eraseOp(nextResetOp);
    } else {
      // Defer this reset into the next one. We want to insert at the
      // front (to keep the order right), so replace this instruction
      // with a new ResetOp
      std::vector<Value> opVec;
      opVec.reserve(resetOp.getNumOperands() + nextResetOp.getNumOperands());
      opVec.insert(opVec.end(), resetOp.getOperands().begin(),
                   resetOp.getOperands().end());
      opVec.insert(opVec.end(), nextResetOp.getOperands().begin(),
                   nextResetOp.getOperands().end());

      rewriter.setInsertionPoint(nextResetOp);
      rewriter.create<ResetQubitOp>(nextResetOp.getLoc(), opVec);
      rewriter.eraseOp(nextResetOp);
      rewriter.eraseOp(resetOp);
    }
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
  // Disable to improve performance
  config.enableRegionSimplification = false;

  patterns.add<MergeResetsTopologicalPattern>(&getContext());

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

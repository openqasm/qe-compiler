//===- MergeParallelResets.cpp - Merge reset ops ----------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
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
// single reset op.
struct MergeParallelResetsPattern : public OpRewritePattern<ResetQubitOp> {

  explicit MergeParallelResetsPattern(MLIRContext *ctx)
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

}; // MergeParallelResetsPattern
} // anonymous namespace

void MergeParallelResetsPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  mlir::GreedyRewriteConfig config;

  // use cheaper top-down traversal (in this case, bottom-up would not behave
  // any differently)
  config.useTopDownTraversal = true;

  patterns.insert<MergeParallelResetsPattern>(&getContext());

  if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config)))
    signalPassFailure();
} // MergeParallelResetsPass::runOnOperation

llvm::StringRef MergeParallelResetsPass::getArgument() const {
  return "merge-resets";
}
llvm::StringRef MergeParallelResetsPass::getDescription() const {
  return "Merge qubit reset ops that can be parallelized into a "
         "single operation.";
}

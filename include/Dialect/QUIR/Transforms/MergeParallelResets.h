//===- MergeParallelResets.h - Merge reset ops ------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for merging resets into a single reset op
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_MERGE_PARALLEL_RESETS_H
#define QUIR_MERGE_PARALLEL_RESETS_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {
/// This pass merges qubit reset operations that can be parallelized into a
/// single reset op lexicographically.
struct MergeResetsLexicographicPass
    : public mlir::PassWrapper<MergeResetsLexicographicPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MergeResetsLexicographicPass() = default;
  MergeResetsLexicographicPass(const MergeResetsLexicographicPass &pass) =
      default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MergeResetsLexicographicPass

/// This pass merges qubit reset operations that can be parallelized into a
/// single reset op topologically.
struct MergeResetsTopologicalPass
    : public mlir::PassWrapper<MergeResetsTopologicalPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MergeResetsTopologicalPass() = default;
  MergeResetsTopologicalPass(const MergeResetsTopologicalPass &pass) = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MergeResetsTopologicalPass

} // namespace mlir::quir

#endif // QUIR_MERGE_PARALLEL_RESETS_H

//===- MergeParallelResets.h - Merge reset ops ------------------*- C++ -*-===//
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
  llvm::StringRef getName() const override;
  std::string passName =
      "Merge Resets Lexicographic Pass (" + getArgument().str() + ")";
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
  llvm::StringRef getName() const override;
  std::string passName =
      "Merge Resets Topological Pass (" + getArgument().str() + ")";
}; // struct MergeResetsTopologicalPass

} // namespace mlir::quir

#endif // QUIR_MERGE_PARALLEL_RESETS_H

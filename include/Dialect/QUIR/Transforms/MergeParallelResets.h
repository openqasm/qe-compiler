//===- MergeParallelResets.h - Merge reset ops ------------------*- C++ -*-===//
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

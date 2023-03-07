//===- MergeMeasures.h - Merge measurement ops ------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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
///  This file declares the pass for merging measurements into a single measure
///  op
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_MERGE_MEASURES_H
#define QUIR_MERGE_MEASURES_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// @brief Merge together measures in a circuit that are lexicographically
/// adjacent into a single variadic measurement.
struct MergeMeasuresLexographicalPass
    : public PassWrapper<MergeMeasuresLexographicalPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MergeMeasuresLexographicalPass

/// @brief Merge together measures in a circuit that are topologically
/// adjacent into a single variadic measurement.
struct MergeMeasuresTopologicalPass
    : public PassWrapper<MergeMeasuresTopologicalPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MergeMeasuresTopologicalPass

} // namespace mlir::quir

#endif // QUIR_MERGE_MEASURES_H

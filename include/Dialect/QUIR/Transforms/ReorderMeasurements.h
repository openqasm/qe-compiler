//===- ReorderMeasurements.h - Move measurement ops later -------*- C++ -*-===//
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
///  This file declares the pass for moving measurements as late as possible
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_REORDER_MEASURES_H
#define QUIR_REORDER_MEASURES_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// @brief Move measures in a circuit to be as late as possible topologically
struct ReorderMeasurementsPass
    : public PassWrapper<ReorderMeasurementsPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct ReorderMeasurementsPass

} // namespace mlir::quir

#endif // QUIR_REORDER_MEASURES_H

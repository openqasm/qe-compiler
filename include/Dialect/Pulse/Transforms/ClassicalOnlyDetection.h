//===- ClassicalOnlyDetection.h - detect pulse ops --------------*- C++ -*-===//
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
/// Defines pass for updating quir.classicalOnly flag based on the presence of
/// Pulse dialect Ops
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_CLASSICAL_ONLY_H
#define PULSE_CLASSICAL_ONLY_H

#include "Dialect/Pulse/IR/PulseOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

// This pass recurses through the IR and detects when scf
// ops use only classical operations. It then applies the classicalOnly
// attribute to all of the scf ops with a value of either true or false
struct ClassicalOnlyDetectionPass
    : public PassWrapper<ClassicalOnlyDetectionPass, OperationPass<>> {
  auto hasPulseSubOps(Operation *inOp) -> bool;
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // end struct ClassicalOnlyDetectionPass

} // namespace mlir::pulse

#endif // PULSE_CLASSICAL_ONLY_H

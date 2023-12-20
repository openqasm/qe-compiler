//===- ClassicalOnlyDetection.h - detect pulse ops --------------*- C++ -*-===//
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
  llvm::StringRef getName() const override;
  std::string passName =
      "Classical Only Detection Pass (" + getArgument().str() + ")";
}; // end struct ClassicalOnlyDetectionPass

} // namespace mlir::pulse

#endif // PULSE_CLASSICAL_ONLY_H

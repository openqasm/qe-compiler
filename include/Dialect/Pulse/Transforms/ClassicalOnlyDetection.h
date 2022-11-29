//===- ClassicalOnlyDetection.h - detect pulse ops --------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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
}; // end struct ClassicalOnlyDetectionPass

} // namespace mlir::pulse

#endif // PULSE_CLASSICAL_ONLY_H

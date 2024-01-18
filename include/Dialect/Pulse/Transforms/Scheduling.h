//===- scheduling.h --- quantum circuits pulse scheduling -------*- C++ -*-===//
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
///  This file implements the pass for scheduling the quantum circuits at pulse
///  level, based on the availability of involved ports
///
//===----------------------------------------------------------------------===//

#ifndef SCHEDULING_PULSE_SEQUENCES_H
#define SCHEDULING_PULSE_SEQUENCES_H

#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir::pulse {

struct QuantumCircuitPulseSchedulingPass
    : public PassWrapper<QuantumCircuitPulseSchedulingPass,
                         OperationPass<ModuleOp>> {
public:
  enum SchedulingMethod { ALAP, ASAP };
  SchedulingMethod SCHEDULING_METHOD = ALAP;
  uint64_t PRE_MEASURE_BUFFER_DELAY = 0;

  // this pass can optionally receive an string specifying the scheduling
  // method; default method is alap scheduling
  QuantumCircuitPulseSchedulingPass() = default;
  QuantumCircuitPulseSchedulingPass(
      const QuantumCircuitPulseSchedulingPass &pass)
      : PassWrapper(pass) {}
  QuantumCircuitPulseSchedulingPass(SchedulingMethod inSchedulingMethod) {
    SCHEDULING_METHOD = inSchedulingMethod;
  }
  QuantumCircuitPulseSchedulingPass(uint64_t inPreMeasureBufferDelay) {
    PRE_MEASURE_BUFFER_DELAY = inPreMeasureBufferDelay;
  }

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;

  // optionally, one can override the scheduling method with this option
  Option<std::string> schedulingMethod{
      *this, "scheduling-method",
      llvm::cl::desc("an string to specify scheduling method"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

private:
  // map to keep track of next availability of ports
  std::map<std::string, int> portNameToNextAvailabilityMap;

  void scheduleAlap(mlir::pulse::CallSequenceOp quantumCircuitCallSequenceOp);
  int getNextAvailableTimeOfPorts(mlir::ArrayAttr ports);
  void updatePortAvailabilityMap(mlir::ArrayAttr ports,
                                 int updatedAvailableTime);
  static mlir::pulse::SequenceOp
  getSequenceOp(mlir::pulse::CallSequenceOp callSequenceOp);
};
} // namespace mlir::pulse

#endif // SCHEDULING_PULSE_SEQUENCES_H

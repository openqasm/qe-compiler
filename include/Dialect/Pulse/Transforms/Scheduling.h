//===- scheduling.h --- scheduling pulse sequences --------------*- C++ -*-===//
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
///  This file implements the pass for scheduling the pulse sequences of quantum
///  gates inside a circuit, based on the availability of involved ports
///
//===----------------------------------------------------------------------===//

#ifndef SCHEDULING_PULSE_SEQUENCES_H
#define SCHEDULING_PULSE_SEQUENCES_H

#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

struct SchedulingPulseSequencesPass
    : public PassWrapper<SchedulingPulseSequencesPass,
                         OperationPass<ModuleOp>> {
  std::string SCHEDULING_METHOD = "alap";

  // this pass can optionally receive an string specifying the scheduling
  // method; default method is alap scheduling
  SchedulingPulseSequencesPass() = default;
  SchedulingPulseSequencesPass(const SchedulingPulseSequencesPass &pass)
      : PassWrapper(pass) {}
  SchedulingPulseSequencesPass(std::string inSchedulingMethod) {
    SCHEDULING_METHOD = std::move(inSchedulingMethod);
  }

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

  // optionally, one can override the scheduling method with this option
  Option<std::string> schedulingMethod{
      *this, "scheduling-method",
      llvm::cl::desc("an string to specify scheduling method"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  // port based alap scheduling
  void scheduleAlap(mlir::pulse::CallSequenceOp mainFuncCallSequenceOp,
                    ModuleOp moduleOp);
  // map to keep track of next availability of ports
  std::map<std::string, int> portNameToNextAvailabilityMap;

  static mlir::pulse::SequenceOp
  getSequenceOp(mlir::pulse::CallSequenceOp callSequenceOp);
};
} // namespace mlir::pulse

#endif // SCHEDULING_PULSE_SEQUENCES_H

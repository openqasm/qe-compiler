//===- Scheduling.cpp ---  quantum circuits pulse scheduling ----*- C++ -*-===//
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

#include "Dialect/Pulse/Transforms/Scheduling.h"
#include "Dialect/QUIR/Utils/Utils.h"

#define DEBUG_TYPE "SchedulingDebug"

using namespace mlir;
using namespace mlir::pulse;

void quantumCircuitPulseSchedulingPass::runOnOperation() {
  // check for command line override of the scheduling method
  if (schedulingMethod.hasValue())
    SCHEDULING_METHOD = schedulingMethod.getValue();

  ModuleOp moduleOp = getOperation();

  // schedule all the quantum circuits which are root call sequence ops
  moduleOp->walk([&](mlir::pulse::CallSequenceOp callSequenceOp) {
    // return if the call sequence op is not a root op
    if (isa<SequenceOp>(callSequenceOp->getParentOp()))
      return;
    assert(SCHEDULING_METHOD == "alap" &&
           "scheduling method not supported currently");
    scheduleAlap(callSequenceOp);
  });
}

void quantumCircuitPulseSchedulingPass::scheduleAlap(
    mlir::pulse::CallSequenceOp quantumCircuitCallSequenceOp) {

  auto quantumCircuitSequenceOp = getSequenceOp(quantumCircuitCallSequenceOp);
  std::string sequenceName = quantumCircuitSequenceOp.sym_name().str();
  LLVM_DEBUG(llvm::dbgs() << "\nscheduling " << sequenceName << "\n");

  int totalDurationOfQuantumCircuitNegative = 0;
  portNameToNextAvailabilityMap.clear();

  // get the MLIR block of the quantum circuit
  auto quantumCircuitSequenceOpBlock = quantumCircuitSequenceOp.body().begin();
  // go over the MLIR operation of the block in reverse order, and find
  // CallSequenceOps, each of which corresponds to a quantum gate. for each
  // CallSequenceOps, we add a timepoint based on the availability of involved
  // ports; timepoints are <=0 because we're walking in reverse order. Note this
  // pass assumes that the operations inside these CallSequenceOps are already
  // scheduled
  for (auto opIt = quantumCircuitSequenceOpBlock->rbegin(),
            opEnd = quantumCircuitSequenceOpBlock->rend();
       opIt != opEnd; ++opIt) {
    auto &op = *opIt;
    if (auto quantumGateCallSequenceOp =
            dyn_cast<mlir::pulse::CallSequenceOp>(op)) {
      // find quantum gate SequenceOp
      auto quantumGateSequenceOp = getSequenceOp(quantumGateCallSequenceOp);
      std::string quantumGateSequenceName =
          quantumGateSequenceOp.sym_name().str();
      LLVM_DEBUG(llvm::dbgs() << "\tprocessing inner sequence "
                              << quantumGateSequenceName << "\n");

      // find ports of the quantum gate SequenceOp
      auto portsOrError =
          PulseOpSchedulingInterface::getPorts(quantumGateSequenceOp);
      if (auto err = portsOrError.takeError()) {
        quantumGateSequenceOp.emitError() << toString(std::move(err));
        signalPassFailure();
      }
      auto ports = portsOrError.get();

      // find duration of the quantum gate callSequenceOp
      llvm::Expected<uint64_t> durOrError =
          quantumGateSequenceOp.getDuration(quantumGateCallSequenceOp);
      if (auto err = durOrError.takeError()) {
        quantumGateSequenceOp.emitError() << toString(std::move(err));
        signalPassFailure();
      }
      uint64_t quantumGateCallSequenceOpDuration = durOrError.get();
      LLVM_DEBUG(llvm::dbgs() << "\t\tduration "
                              << quantumGateCallSequenceOpDuration << "\n");

      // find next available time for all the ports
      int nextAvailableTimeOfAllPorts = getNextAvailableTimeOfPorts(ports);
      LLVM_DEBUG(llvm::dbgs() << "\t\tnext availability is at "
                              << nextAvailableTimeOfAllPorts << "\n");

      // find the updated available time, i.e., when the current quantum gate
      // will be scheduled
      int updatedAvailableTime =
          nextAvailableTimeOfAllPorts - quantumGateCallSequenceOpDuration;
      LLVM_DEBUG(llvm::dbgs() << "\t\tcurrent gate scheduled at "
                              << updatedAvailableTime << "\n");
      // update the port availability map
      updatePortAvailabilityMap(ports, updatedAvailableTime);

      // keep track of total duration of the quantum circuit
      if (updatedAvailableTime < totalDurationOfQuantumCircuitNegative)
        totalDurationOfQuantumCircuitNegative = updatedAvailableTime;

      // set the timepoint of quantum gate
      PulseOpSchedulingInterface::setTimepoint(quantumGateCallSequenceOp,
                                               updatedAvailableTime);
    }
  }

  // multiply by -1 so that quantum circuit duration becomes positive
  int totalDurationOfQuantumCircuit = -totalDurationOfQuantumCircuitNegative;
  LLVM_DEBUG(llvm::dbgs() << "\ttotal duration of quantum circuit "
                          << totalDurationOfQuantumCircuit << "\n");

  // setting duration of the quantum circuit
  PulseOpSchedulingInterface::setDuration(quantumCircuitSequenceOp,
                                          totalDurationOfQuantumCircuit);
  // setting timepoint of the quantum circuit; at this point, we can add
  // totalDurationOfQuantumCircuit to above <=0 timepoints, so that they become
  // >=0, however, that would require walking the IR again. Instead, we add a
  // postive timepoint to the parent op, i.e., quantum circuit sequence op, and
  // later passes would need to add this value as an offset to determine the
  // effective timepoints
  PulseOpSchedulingInterface::setTimepoint(quantumCircuitSequenceOp,
                                           totalDurationOfQuantumCircuit);
}

int quantumCircuitPulseSchedulingPass::getNextAvailableTimeOfPorts(
    mlir::ArrayAttr ports) {
  int nextAvailableTimeOfAllPorts = 0;
  for (auto attr : ports) {
    std::string portName = attr.dyn_cast<StringAttr>().getValue().str();
    if (portName.empty())
      continue;
    if (portNameToNextAvailabilityMap.find(portName) !=
        portNameToNextAvailabilityMap.end()) {
      if (portNameToNextAvailabilityMap[portName] < nextAvailableTimeOfAllPorts)
        nextAvailableTimeOfAllPorts = portNameToNextAvailabilityMap[portName];
    }
  }
  return nextAvailableTimeOfAllPorts;
}

void quantumCircuitPulseSchedulingPass::updatePortAvailabilityMap(
    mlir::ArrayAttr ports, int updatedAvailableTime) {
  for (auto attr : ports) {
    std::string portName = attr.dyn_cast<StringAttr>().getValue().str();
    if (portName.empty())
      continue;
    portNameToNextAvailabilityMap[portName] = updatedAvailableTime;
  }
}

mlir::pulse::SequenceOp quantumCircuitPulseSchedulingPass::getSequenceOp(
    mlir::pulse::CallSequenceOp callSequenceOp) {
  auto seqAttr = callSequenceOp->getAttrOfType<FlatSymbolRefAttr>("callee");
  assert(seqAttr && "Requires a 'callee' symbol reference attribute");

  auto sequenceOp =
      SymbolTable::lookupNearestSymbolFrom<mlir::pulse::SequenceOp>(
          callSequenceOp, seqAttr);
  assert(sequenceOp && "matching sequence not found");
  return sequenceOp;
}

llvm::StringRef quantumCircuitPulseSchedulingPass::getArgument() const {
  return "quantum-circuit-pulse-scheduling";
}

llvm::StringRef quantumCircuitPulseSchedulingPass::getDescription() const {
  return "Scheduling a quantum circuit at pulse level.";
}

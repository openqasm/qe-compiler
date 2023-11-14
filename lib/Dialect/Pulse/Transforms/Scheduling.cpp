//===- scheduling.cpp --- scheduling pulse sequences ------------*- C++ -*-===//
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

#include "Dialect/Pulse/Transforms/Scheduling.h"
#include "Dialect/QUIR/Utils/Utils.h"

#define DEBUG_TYPE "SchedulingDebug"

using namespace mlir;
using namespace mlir::pulse;

void SchedulingPulseSequencesPass::runOnOperation() {
  // check for command line override of the scheduling method
  if (schedulingMethod.hasValue())
    SCHEDULING_METHOD = schedulingMethod.getValue();

  ModuleOp moduleOp = getOperation();
  FuncOp mainFunc = dyn_cast<FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");

  // for each pulse sequence, schedule the inner call sequences
  mainFunc->walk([&](mlir::pulse::CallSequenceOp mainFuncCallSequenceOp) {
    assert(SCHEDULING_METHOD == "alap" &&
           "scheduling method not supported currently");
    scheduleAlap(mainFuncCallSequenceOp, moduleOp);
  });
}

void SchedulingPulseSequencesPass::scheduleAlap(
    mlir::pulse::CallSequenceOp mainFuncCallSequenceOp, ModuleOp moduleOp) {
  mlir::OpBuilder builder(moduleOp);

  auto mainFuncSequenceOp = getSequenceOp(mainFuncCallSequenceOp);
  std::string sequenceName = mainFuncSequenceOp.sym_name().str();
  LLVM_DEBUG(llvm::dbgs() << "\nscheduling " << sequenceName << "\n");

  int duration = 0;
  portNameToNextAvailabilityMap.clear();
  for (auto &block : mainFuncSequenceOp) {
    for (auto opIt = block.rbegin(), opEnd = block.rend(); opIt != opEnd;
         ++opIt) {
      auto &op = *opIt;
      if (auto innerCallSequenceOp =
              dyn_cast<mlir::pulse::CallSequenceOp>(op)) {
        // find innerSequenceOp
        auto innerSequenceOp = getSequenceOp(innerCallSequenceOp);
        std::string innerSequenceName = innerSequenceOp.sym_name().str();
        LLVM_DEBUG(llvm::dbgs() << "\tprocessing inner sequence "
                                << innerSequenceName << "\n");

        // find arg ports
        assert(innerSequenceOp->hasAttrOfType<ArrayAttr>("pulse.argPorts") and
               "no pulse.argPorts found for the innerSequenceOp.");
        auto argPortsAttr =
            innerSequenceOp->getAttrOfType<ArrayAttr>("pulse.argPorts");

        // find duration of the inner callseq
        uint innerCallSequenceDuration = 0;
        if (innerCallSequenceOp->hasAttrOfType<IntegerAttr>("pulse.duration"))
          innerCallSequenceDuration = static_cast<uint64_t>(
              innerCallSequenceOp->getAttrOfType<IntegerAttr>("pulse.duration")
                  .getInt());
        else if (innerSequenceOp->hasAttrOfType<IntegerAttr>("pulse.duration"))
          innerCallSequenceDuration = static_cast<uint64_t>(
              innerSequenceOp->getAttrOfType<IntegerAttr>("pulse.duration")
                  .getInt());
        else
          llvm_unreachable("no pulse.duration specified");
        LLVM_DEBUG(llvm::dbgs()
                   << "\t\tduration " << innerCallSequenceDuration << "\n");

        // find next avail
        int nextAvailOfAllPorts = 0;
        for (auto attr : argPortsAttr) {
          std::string portName = attr.dyn_cast<StringAttr>().getValue().str();
          if (portName.empty())
            continue;
          if (portNameToNextAvailabilityMap.find(portName) !=
              portNameToNextAvailabilityMap.end()) {
            if (portNameToNextAvailabilityMap[portName] < nextAvailOfAllPorts)
              nextAvailOfAllPorts = portNameToNextAvailabilityMap[portName];
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "\t\tnext availability is at "
                                << nextAvailOfAllPorts << "\n");

        // update the next availability map
        int startingTime = nextAvailOfAllPorts - innerCallSequenceDuration;
        LLVM_DEBUG(llvm::dbgs() << "\t\tscheduled at " << startingTime << "\n");
        if (startingTime < duration)
          duration = startingTime;
        for (auto attr : argPortsAttr) {
          std::string portName = attr.dyn_cast<StringAttr>().getValue().str();
          if (portName.empty())
            continue;
          portNameToNextAvailabilityMap[portName] = startingTime;
        }

        // set startingTime for innerCallSequenceOp
        PulseOpSchedulingInterface::setTimepoint(innerCallSequenceOp,
                                                 startingTime);
      }
    }
  }

  // multiply by -1 so that duration becomes positive
  duration = -duration;
  LLVM_DEBUG(llvm::dbgs() << "\ttotal circuit duration " << duration << "\n");
  // setting timepoint and duration for mainFuncSequenceOp
  PulseOpSchedulingInterface::setTimepoint(mainFuncSequenceOp, duration);
  PulseOpSchedulingInterface::setDuration(mainFuncSequenceOp, duration);
}

mlir::pulse::SequenceOp SchedulingPulseSequencesPass::getSequenceOp(
    mlir::pulse::CallSequenceOp callSequenceOp) {
  auto seqAttr = callSequenceOp->getAttrOfType<FlatSymbolRefAttr>("callee");
  assert(seqAttr && "Requires a 'callee' symbol reference attribute");

  auto sequenceOp =
      SymbolTable::lookupNearestSymbolFrom<mlir::pulse::SequenceOp>(
          callSequenceOp, seqAttr);
  assert(sequenceOp && "matching sequence not found");
  return sequenceOp;
}

llvm::StringRef SchedulingPulseSequencesPass::getArgument() const {
  return "scheduling-pulse-sequences";
}

llvm::StringRef SchedulingPulseSequencesPass::getDescription() const {
  return "Scheduling the pulse sequences of the quantum gates inside a "
         "circuit.";
}

//===- Scheduling.cpp ---  quantum circuits pulse scheduling ----*- C++ -*-===//
//
// (C) Copyright IBM 2023, 2024.
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
///  level, based on the availability of involved mix frames
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/Scheduling.h"
#include "Dialect/Pulse/IR/PulseInterfaces.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"
#include "Utils/SymbolCacheAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define DEBUG_TYPE "SchedulingDebug"

using namespace mlir;
using namespace mlir::pulse;

void QuantumCircuitPulseSchedulingPass::runOnOperation() {
  // check for command line override of the scheduling method
  if (schedulingMethod.hasValue()) {
    if (schedulingMethod.getValue() == "alap")
      SCHEDULING_METHOD = ALAP;
    else if (schedulingMethod.getValue() == "asap")
      SCHEDULING_METHOD = ASAP;
    else
      llvm_unreachable("scheduling method not supported currently");
  }

  // check for command line override of the pre measure buffer delay
  if (preMeasureBufferDelay.hasValue())
    PRE_MEASURE_BUFFER_DELAY = preMeasureBufferDelay.getValue();

  ModuleOp const moduleOp = getOperation();

  // populate/cache the symbol map
  symbolCache = &getAnalysis<qssc::utils::SymbolCacheAnalysis>()
                     .invalidate()
                     .addToCache<SequenceOp>();

  // schedule all the quantum circuits which are root call sequence ops
  moduleOp->walk([&](mlir::pulse::CallSequenceOp callSequenceOp) {
    // return if the call sequence op is not a root op
    if (isa<SequenceOp>(callSequenceOp->getParentOp()))
      return;
    switch (SCHEDULING_METHOD) {
    case ALAP:
      scheduleAlap(callSequenceOp);
      break;
    default:
      llvm_unreachable("scheduling method not supported currently");
    }
  });
}

void QuantumCircuitPulseSchedulingPass::scheduleAlap(
    mlir::pulse::CallSequenceOp quantumCircuitCallSequenceOp) {

  assert(symbolCache && "symbolCache not set");
  auto quantumCircuitSequenceOp =
      symbolCache->getOp<SequenceOp>(quantumCircuitCallSequenceOp);
  std::string const sequenceName = quantumCircuitSequenceOp.getSymName().str();
  LLVM_DEBUG(llvm::dbgs() << "\nscheduling " << sequenceName << "\n");

  int64_t totalDurationOfQuantumCircuitNegative = 0;
  mixFrameToNextAvailabilityMap.clear();

  // get the MLIR block of the quantum circuit
  auto quantumCircuitSequenceOpBlock =
      quantumCircuitSequenceOp.getBody().begin();
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
      assert(symbolCache && "symbolCache not set");
      auto quantumGateSequenceOp =
          symbolCache->getOp<SequenceOp>(quantumGateCallSequenceOp);
      const std::string quantumGateSequenceName =
          quantumGateSequenceOp.getSymName().str();
      LLVM_DEBUG(llvm::dbgs() << "\tprocessing inner sequence "
                              << quantumGateSequenceName << "\n");

      // find the block argument numbers of the mix frames
      std::unordered_set<unsigned int> mixFramesBlockArgNums =
          getMixFramesBlockArgNums(quantumGateCallSequenceOp);

      // find duration of the quantum gate callSequenceOp
      llvm::Expected<uint64_t> durOrError =
          quantumGateSequenceOp.getDuration(quantumGateCallSequenceOp);
      if (auto err = durOrError.takeError()) {
        quantumGateSequenceOp.emitError() << toString(std::move(err));
        signalPassFailure();
      }
      const uint64_t quantumGateCallSequenceOpDuration = durOrError.get();
      LLVM_DEBUG(llvm::dbgs() << "\t\tduration "
                              << quantumGateCallSequenceOpDuration << "\n");

      // find next available time for all the mix frames
      const int64_t nextAvailableTimeOfAllMixFrames =
          getNextAvailableTimeOfMixFrames(mixFramesBlockArgNums);
      LLVM_DEBUG(llvm::dbgs() << "\t\tnext availability is at "
                              << nextAvailableTimeOfAllMixFrames << "\n");

      // find the updated available time, i.e., when the current quantum gate
      // will be scheduled
      int64_t updatedAvailableTime =
          nextAvailableTimeOfAllMixFrames - quantumGateCallSequenceOpDuration;
      // set the timepoint of quantum gate
      PulseOpSchedulingInterface::setTimepoint(quantumGateCallSequenceOp,
                                               updatedAvailableTime);
      LLVM_DEBUG(llvm::dbgs() << "\t\tcurrent gate scheduled at "
                              << updatedAvailableTime << "\n");
      // update the mix frame availability map
      if (sequenceOpIncludeCapture(quantumGateSequenceOp))
        updatedAvailableTime -= PRE_MEASURE_BUFFER_DELAY;
      updateMixFrameAvailabilityMap(mixFramesBlockArgNums,
                                    updatedAvailableTime);

      // keep track of total duration of the quantum circuit
      if (updatedAvailableTime < totalDurationOfQuantumCircuitNegative)
        totalDurationOfQuantumCircuitNegative = updatedAvailableTime;
    }
  }

  // multiply by -1 so that quantum circuit duration becomes positive
  const int64_t totalDurationOfQuantumCircuit =
      -totalDurationOfQuantumCircuitNegative;
  LLVM_DEBUG(llvm::dbgs() << "\ttotal duration of quantum circuit "
                          << totalDurationOfQuantumCircuit << "\n");

  // setting duration of the quantum call circuit
  PulseOpSchedulingInterface::setDuration(quantumCircuitCallSequenceOp,
                                          totalDurationOfQuantumCircuit);
  // setting timepoint of the quantum call circuit; at this point, we can add
  // totalDurationOfQuantumCircuit to above <=0 timepoints, so that they become
  // >=0, however, that would require walking the IR again. Instead, we add a
  // postive timepoint to the parent op, i.e., quantum circuit call sequence op,
  // and later passes would need to add this value as an offset to determine the
  // effective timepoints
  PulseOpSchedulingInterface::setTimepoint(quantumCircuitCallSequenceOp,
                                           totalDurationOfQuantumCircuit);
}

int64_t QuantumCircuitPulseSchedulingPass::getNextAvailableTimeOfMixFrames(
    std::unordered_set<unsigned int> &mixFramesBlockArgNums) {
  int64_t nextAvailableTimeOfAllMixFrames = 0;
  for (auto id : mixFramesBlockArgNums)
    if (mixFrameToNextAvailabilityMap.find(id) !=
            mixFrameToNextAvailabilityMap.end() &&
        mixFrameToNextAvailabilityMap[id] < nextAvailableTimeOfAllMixFrames)
      nextAvailableTimeOfAllMixFrames = mixFrameToNextAvailabilityMap[id];
  return nextAvailableTimeOfAllMixFrames;
}

std::unordered_set<unsigned int>
QuantumCircuitPulseSchedulingPass::getMixFramesBlockArgNums(
    mlir::pulse::CallSequenceOp quantumGateCallSequenceOp) {
  std::unordered_set<unsigned int> mixFramesBlockArgNums;
  for (auto const &argumentResult :
       llvm::enumerate(quantumGateCallSequenceOp.getOperands())) {
    auto argType = argumentResult.value().getType();
    if (auto mixFrameType = argType.dyn_cast<mlir::pulse::MixedFrameType>()) {
      mixFramesBlockArgNums.insert(
          argumentResult.value().dyn_cast<BlockArgument>().getArgNumber());
    }
  }
  return mixFramesBlockArgNums;
}

void QuantumCircuitPulseSchedulingPass::updateMixFrameAvailabilityMap(
    std::unordered_set<unsigned int> &mixFramesBlockArgNums,
    int64_t updatedAvailableTime) {
  for (auto id : mixFramesBlockArgNums)
    mixFrameToNextAvailabilityMap[id] = updatedAvailableTime;
}

bool QuantumCircuitPulseSchedulingPass::sequenceOpIncludeCapture(
    mlir::pulse::SequenceOp quantumGateSequenceOp) {
  bool sequenceOpIncludeCapture = false;
  quantumGateSequenceOp->walk([&](mlir::pulse::CaptureOp op) {
    sequenceOpIncludeCapture = true;
    return WalkResult::interrupt();
  });

  return sequenceOpIncludeCapture;
}

llvm::StringRef QuantumCircuitPulseSchedulingPass::getArgument() const {
  return "quantum-circuit-pulse-scheduling";
}

llvm::StringRef QuantumCircuitPulseSchedulingPass::getDescription() const {
  return "Scheduling a quantum circuit at pulse level.";
}

llvm::StringRef QuantumCircuitPulseSchedulingPass::getName() const {
  return "Quantum Circuit Pulse Scheduling Pass";
}

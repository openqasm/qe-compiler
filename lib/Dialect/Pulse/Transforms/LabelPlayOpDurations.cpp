//===- LabelPlayOpDurations.cpp - Label PlayOps with Durations --*- C++ -*-===//
//
// (C) Copyright IBM 2024.
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
///  This file implements the pass for labeling pulse.play operations with the
///  duration of the waveform being played.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/LabelPlayOpDurations.h"
#include "Dialect/Pulse/IR/PulseInterfaces.h"
#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;
using namespace mlir::pulse;

void LabelPlayOpDurationsPass::runOnOperation() {

  // all PlayOps are assumed to be inside of a pulse.sequence
  // pass builds a mapping of sequence name , argument number to duration
  // for all play operations using call_sequences
  //
  // pass then searches for all play operations and assigns the durations using
  // the mapping

  Operation *module = getOperation();

  std::unordered_map<std::string, std::vector<uint64_t>> argumentToDuration;

  module->walk([&](CallSequenceOp callSequenceOp) {
    auto callee = callSequenceOp.getCallee().str();

    for (const auto &operand : callSequenceOp->getOperands()) {

      uint64_t duration = 0;
      auto *defOp = operand.getDefiningOp();
      if (defOp)
        if (auto waveformOp = dyn_cast<Waveform_CreateOp>(defOp))
          duration = waveformOp.getDuration(nullptr /*callSequenceOp*/).get();

      argumentToDuration[callee].push_back(duration);
    }
  });

  module->walk([&](PlayOp playOp) {
    auto sequenceOp = playOp->getParentOfType<mlir::pulse::SequenceOp>();
    auto sequenceStr = sequenceOp.getSymName().str();
    // sequenceStr may not be in argumentToDuration if the sequence is not
    // actually called
    auto searchSequence = argumentToDuration.find(sequenceStr);
    if (searchSequence == argumentToDuration.end())
      return;
    auto wfArgNumber = playOp.getWfr().dyn_cast<BlockArgument>().getArgNumber();
    auto duration = searchSequence->second[wfArgNumber];
    mlir::pulse::PulseOpSchedulingInterface::setDuration(playOp, duration);
  });

} // runOnOperation

llvm::StringRef LabelPlayOpDurationsPass::getArgument() const {
  return "pulse-label-play-op-duration";
}

llvm::StringRef LabelPlayOpDurationsPass::getDescription() const {
  return "Label PlayOps with duration attributes";
}

llvm::StringRef LabelPlayOpDurationsPass::getName() const {
  return "Label PlayOp Durations Pass";
}

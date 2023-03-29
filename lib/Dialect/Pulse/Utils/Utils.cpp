//===- Utils.cpp - Pulse Utilities ------------------------------*- C++ -*-===//
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
///  This file implements some utility functions for Pulse passes
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Utils/Utils.h"
#include "Dialect/Pulse/IR/PulseOps.h"

namespace mlir::pulse {

// TODO: update this function based on the config files
uint getQubitId(Port_CreateOp pulsePortOp) {
  auto portNumId = pulsePortOp.uid().drop_front();
  uint idNum = 0;
  portNumId.getAsInteger(0, idNum);
  return idNum;
}

uint getQubitId(MixFrameOp mixFrameOp) {
  // get qubit id from a MixFrameOp
  auto portOp = mixFrameOp.port().getDefiningOp<mlir::pulse::Port_CreateOp>();
  return getQubitId(portOp);
}

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceOp callSequenceOp) {
  auto wfrArgIndex = pulsePlayOp.wfr().dyn_cast<BlockArgument>().getArgNumber();
  auto wfrOp = callSequenceOp.getOperand(wfrArgIndex)
                   .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
  return wfrOp;
}

bool isPlayOpForDrive(Operation *op,
                      mlir::pulse::CallSequenceOp callSequenceOp) {

  auto pulsePlayOp = dyn_cast<mlir::pulse::PlayOp>(op);
  auto mixFrameArgIndex =
      pulsePlayOp.target().dyn_cast<BlockArgument>().getArgNumber();
  auto mixFrameOp = callSequenceOp.getOperand(mixFrameArgIndex)
                        .getDefiningOp<mlir::pulse::MixFrameOp>();

  if (mixFrameOp.signalType().str() == "drive")
    return true;

  return false;
}

int getTimepoint(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("pulse.timepoint").getInt();
}

} // end namespace mlir::pulse

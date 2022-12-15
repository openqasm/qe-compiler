//===- Utils.cpp - Pulse Utilities ------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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
#include "Dialect/Pulse/PulseOps.h"

namespace mlir::pulse {

// TODO: update this function based on the config files
uint getQubitId(Port_CreateOp &pulsePortOp) {
  auto portNumId = pulsePortOp.uid().drop_front();
  uint idNum = 0;
  portNumId.getAsInteger(0, idNum);
  return idNum;
}

uint getQubitId(PlayOp &pulsePlayOp, CallSequenceOp &callSequenceOp) {
  // get qubit id from a PlayOp under the assumption that the play op
  // is part of a sequence
  auto mixFrameOp = getMixFrameOp(pulsePlayOp, callSequenceOp);
  auto portOp = mixFrameOp.port().getDefiningOp<mlir::pulse::Port_CreateOp>();
  return getQubitId(portOp);
}

uint getQubitId(CaptureOp &captureOp, CallSequenceOp &callSequenceOp) {
  // get qubit id from a CaptureOp under the assumption that the capture op
  // is part of a sequence
  auto portOpArgIndex =
      captureOp.target().dyn_cast<BlockArgument>().getArgNumber();
  auto mixFrameOp = callSequenceOp.getOperand(portOpArgIndex)
                        .getDefiningOp<mlir::pulse::MixFrameOp>();
  auto portOp =
      dyn_cast<mlir::pulse::Port_CreateOp>(mixFrameOp.port().getDefiningOp());
  return getQubitId(portOp);
}

Waveform_CreateOp getWaveformOp(PlayOp &pulsePlayOp,
                                CallSequenceOp &callSequenceOp) {
  auto wfrArgIndex = pulsePlayOp.wfr().dyn_cast<BlockArgument>().getArgNumber();
  auto wfrOp = callSequenceOp.getOperand(wfrArgIndex)
                   .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
  return wfrOp;
}

MixFrameOp getMixFrameOp(PlayOp &pulsePlayOp, CallSequenceOp &callSequenceOp) {
  auto mixFrameArgIndex =
      pulsePlayOp.target().dyn_cast<BlockArgument>().getArgNumber();
  auto mixFrameOp = callSequenceOp.getOperand(mixFrameArgIndex)
                        .getDefiningOp<mlir::pulse::MixFrameOp>();
  return mixFrameOp;
}

bool isPlayOpForDrive(Operation *op,
                      mlir::pulse::CallSequenceOp &callSequenceOp) {

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

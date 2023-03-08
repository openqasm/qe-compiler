//===- Utils.cpp - Pulse Utilities ------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

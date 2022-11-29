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

namespace mlir::pulse {

// TODO: update this function based on the config files
uint getQubitId(Port_CreateOp *pulsePortOp) {
  auto portNumId = pulsePortOp->uid().drop_front();
  uint idNum = 0;
  portNumId.getAsInteger(0, idNum);
  return idNum;
}

bool isPlayOpForDrive(Operation *op,
                      mlir::pulse::CallSequenceOp &callSequenceOp) {

  auto pulsePlayOp = dyn_cast<mlir::pulse::PlayOp>(op);
  auto mixFrameArgIndex =
      pulsePlayOp.target().dyn_cast<BlockArgument>().getArgNumber();
  auto mixFrameOp = callSequenceOp.getOperand(mixFrameArgIndex)
                        .getDefiningOp<mlir::pulse::MixFrameOp>();

  if (mixFrameOp.signal_type().str() == "drive")
    return true;

  return false;
}

int getTimepoint(Operation *op) {
  return op->getAttrOfType<IntegerAttr>(timepointAttrName).getInt();
}

} // end namespace mlir::pulse

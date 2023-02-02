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
#include "Dialect/Pulse/IR/PulseOps.h"

namespace mlir::pulse {

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceOp callSequenceOp) {
  auto wfrArgIndex = pulsePlayOp.wfr().dyn_cast<BlockArgument>().getArgNumber();
  auto wfrOp = callSequenceOp.getOperand(wfrArgIndex)
                   .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
  return wfrOp;
}

int getTimepoint(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("pulse.timepoint").getInt();
}

} // end namespace mlir::pulse

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

#include "mlir/IR/Value.h"

namespace mlir::pulse {

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceOp callSequenceOp) {
  auto wfrArgIndex =
      pulsePlayOp.getWfr().dyn_cast<BlockArgument>().getArgNumber();
  auto wfrOp = callSequenceOp.getOperand(wfrArgIndex)
                   .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
  return wfrOp;
}

Waveform_CreateOp
getWaveformOp(PlayOp pulsePlayOp,
              std::deque<mlir::pulse::CallSequenceOp> &callSequenceOpStack) {
  auto wfrIndex = 0;
  mlir::Value wfrOp = pulsePlayOp.getWfr();

  for (auto it = callSequenceOpStack.rbegin(); it != callSequenceOpStack.rend();
       ++it) {
    wfrIndex = wfrOp.dyn_cast<BlockArgument>().getArgNumber();
    wfrOp = it->getOperand(wfrIndex);
  }

  auto waveformOp =
      dyn_cast<mlir::pulse::Waveform_CreateOp>(wfrOp.getDefiningOp());
  if (!waveformOp)
    pulsePlayOp->emitError() << "The wfr argument is not a Waveform_CreateOp.";
  return waveformOp;
}

double
getPhaseValue(ShiftPhaseOp shiftPhaseOp,
              std::deque<mlir::pulse::CallSequenceOp> &callSequenceOpStack) {
  auto phaseOffsetIndex = 0;
  mlir::Value phaseOffset = shiftPhaseOp.getPhaseOffset();

  for (auto it = callSequenceOpStack.rbegin(); it != callSequenceOpStack.rend();
       ++it) {
    if (phaseOffset.isa<BlockArgument>()) {
      phaseOffsetIndex = phaseOffset.dyn_cast<BlockArgument>().getArgNumber();
      phaseOffset = it->getOperand(phaseOffsetIndex);
    } else
      break;
  }

  auto phaseOffsetOp =
      dyn_cast<mlir::arith::ConstantFloatOp>(phaseOffset.getDefiningOp());
  assert(phaseOffsetOp && "phase offset is not a ConstantFloatOp");
  return phaseOffsetOp.value().convertToDouble();
}

} // end namespace mlir::pulse

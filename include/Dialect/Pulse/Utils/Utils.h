//===- Utils.h - Pulse Utilities --------------------------------*- C++ -*-===//
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
///  This file declares some utility functions for Pulse passes
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringRef.h"

#include <deque>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

using CallSequenceStack_t = std::deque<mlir::pulse::CallSequenceOp>;

namespace mlir::pulse {

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceOp callSequenceOp);

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceStack_t &callSequenceOpStack);

double getPhaseValue(ShiftPhaseOp shiftPhaseOp,
                     CallSequenceStack_t &callSequenceOpStack);

/// this function goes over all the blocks of the input pulse sequence, and for
/// each block, it sorts the pulse ops within the block according to their
/// timepoints.
void sortOpsByTimepoint(SequenceOp &sequenceOp);

template <typename PulseOpTy>
MixFrameOp getMixFrameOp(PulseOpTy pulseOp, CallSequenceOp callSequenceOp) {

  auto frameArgIndex =
      pulseOp.getTarget().template cast<BlockArgument>().getArgNumber();
  auto frameOp = callSequenceOp.getOperand(frameArgIndex).getDefiningOp();

  auto mixFrameOp = dyn_cast<mlir::pulse::MixFrameOp>(frameOp);

  if (!mixFrameOp)
    pulseOp->emitError() << "The target argument is not a MixFrameOp.";
  return mixFrameOp;
}

template <typename PulseOpTy>
MixFrameOp getMixFrameOp(PulseOpTy pulseOp,
                         CallSequenceStack_t &callSequenceOpStack) {

  auto targetIndex = 0;
  auto target = pulseOp.getTarget();

  for (auto it = callSequenceOpStack.rbegin(); it != callSequenceOpStack.rend();
       ++it) {
    targetIndex = target.template cast<BlockArgument>().getArgNumber();
    target = it->getOperand(targetIndex);
  }

  auto mixFrameOp = dyn_cast<mlir::pulse::MixFrameOp>(target.getDefiningOp());
  if (!mixFrameOp)
    pulseOp->emitError() << "The target argument is not a MixFrameOp.";
  return mixFrameOp;
}

} // end namespace mlir::pulse

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

#include "Dialect/Pulse/IR/PulseInterfaces.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTraits.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <optional>

namespace mlir::pulse {

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceOp callSequenceOp) {
  auto wfrArgIndex =
      pulsePlayOp.getWfr().dyn_cast<BlockArgument>().getArgNumber();
  auto wfrOp = callSequenceOp.getOperand(wfrArgIndex)
                   .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
  return wfrOp;
}

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceStack_t &callSequenceOpStack) {
  auto wfrIndex = 0;
  mlir::Value wfrOp = pulsePlayOp.getWfr();

  for (auto it = callSequenceOpStack.rbegin(); it != callSequenceOpStack.rend();
       ++it) {
    if (wfrOp.isa<BlockArgument>()) {
      wfrIndex = wfrOp.dyn_cast<BlockArgument>().getArgNumber();
      wfrOp = it->getOperand(wfrIndex);
    } else
      break;
  }

  auto waveformOp = dyn_cast<Waveform_CreateOp>(wfrOp.getDefiningOp());
  if (!waveformOp)
    pulsePlayOp->emitError() << "The wfr argument is not a Waveform_CreateOp.";
  return waveformOp;
}

double getPhaseValue(ShiftPhaseOp shiftPhaseOp,
                     CallSequenceStack_t &callSequenceOpStack) {
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
  if (!phaseOffsetOp)
    phaseOffsetOp->emitError() << "Phase offset is not a ConstantFloatOp.";
  return phaseOffsetOp.value().convertToDouble();
}

void sortOpsByTimepoint(SequenceOp &sequenceOp) {
  // sort ops by timepoint
  for (Region &region : sequenceOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      auto &blockOps = block.getOperations();
      blockOps.sort(
          [&](Operation &op1, Operation &op2) {
            // put constants ahead of everything else
            if (isa<arith::ConstantOp>(op1) && !isa<arith::ConstantOp>(op2))
              return true;

            bool const testOp1 = (op1.hasTrait<mlir::pulse::HasTargetFrame>() ||
                                  isa<CallSequenceOp>(op1));
            bool const testOp2 = (op2.hasTrait<mlir::pulse::HasTargetFrame>() ||
                                  isa<CallSequenceOp>(op2));

            if (!testOp1 || !testOp2)
              return false;

            std::optional<int64_t> currentTimepoint =
                PulseOpSchedulingInterface::getTimepoint(&op1);
            if (!currentTimepoint.has_value()) {
              op1.emitError()
                  << "Operation does not have a pulse.timepoint attribute.";
            }
            std::optional<int64_t> nextTimepoint =
                PulseOpSchedulingInterface::getTimepoint(&op2);
            if (!nextTimepoint.has_value()) {
              op2.emitError()
                  << "Operation does not have a pulse.timepoint attribute.";
            }

            if (currentTimepoint.value() == nextTimepoint.value()) {
              // if timepoints are equal, put non-playOp/captureOp (e.g.,
              // shiftPhaseOp) ahead of playOp/captureOp
              if (isa<PlayOp>(op1) or isa<CaptureOp>(op1))
                return false;
              return true;
            }
            // order by timepoint
            return currentTimepoint.value() < nextTimepoint.value();
          }); // blockOps.sort
    }
  }
}

} // end namespace mlir::pulse

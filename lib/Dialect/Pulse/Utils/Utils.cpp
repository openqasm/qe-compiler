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

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceOp callSequenceOp) {
  auto wfrArgIndex = pulsePlayOp.getWfr().dyn_cast<BlockArgument>().getArgNumber();
  auto wfrOp = callSequenceOp.getOperand(wfrArgIndex)
                   .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
  return wfrOp;
}

} // end namespace mlir::pulse

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
///  This file defines the pass for labeling pulse.play operations with the
///  duration of the waveform being played.
//===----------------------------------------------------------------------===//

#ifndef PULSE_LABEL_PLAY_DURATION_H
#define PULSE_LABEL_PLAY_DURATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

class LabelPlayOpDurationsPass
    : public PassWrapper<LabelPlayOpDurationsPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
};
} // namespace mlir::pulse

#endif // PULSE_LABEL_PLAY_DURATION_H

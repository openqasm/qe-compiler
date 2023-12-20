//===- MergeDelays.h - Merge back to back pulse.delays  ---------*- C++ -*-===//
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
///  This file declares the pass for merging back to back pulse.delays.
///  The current implementation defaults to ignoring the target, there
///  is an option to override this.
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_MERGE_DELAYS_H
#define PULSE_MERGE_DELAYS_H

#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

class MergeDelayPass
    : public PassWrapper<MergeDelayPass, OperationPass<SequenceOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName = 
      "Merge Delay Pass (" + getArgument().str() + ")";
};
} // namespace mlir::pulse

#endif // PULSE_MERGE_DELAY_H

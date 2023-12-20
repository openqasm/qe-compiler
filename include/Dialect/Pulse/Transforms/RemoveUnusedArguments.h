//===- RemoveUnusedArguments.h - remove unused args from call ---*- C++ -*-===//
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
///  This file declares the pass for removing arguments from pulse.call_sequence
///  where the argument is unused in the callee pulse.sequence.
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_REMOVE_UNUSED_ARGUMENTS_H
#define PULSE_REMOVE_UNUSED_ARGUMENTS_H

#include <deque>
#include <set>

#include "Dialect/Pulse/IR/PulseOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

class RemoveUnusedArgumentsPass
    : public PassWrapper<RemoveUnusedArgumentsPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName =
      "Remove Unused Arguments Pass (" + getArgument().str() + ")";
};
} // namespace mlir::pulse

#endif // PULSE_REMOVE_UNUSED_ARGUMENTS_H

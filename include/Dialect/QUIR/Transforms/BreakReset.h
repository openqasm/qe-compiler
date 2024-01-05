//===- BreakReset.h - Breset reset ops --------------------------*- C++ -*-===//
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
///  This file declares the pass for breaking resets into control-flow.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_BREAK_RESET_H
#define QUIR_BREAK_RESET_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// This pass converts input ResetQubitOps to a parameterized number of
/// measure and the conditionally called x gates, as well as an optional delay.
struct BreakResetPass
    : public mlir::PassWrapper<BreakResetPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  BreakResetPass() = default;
  BreakResetPass(const BreakResetPass &pass) : PassWrapper(pass) {}
  BreakResetPass(uint inNumIterations, uint inDelayCycles) {
    numIterations = inNumIterations;
    delayCycles = inDelayCycles;
  }

  void runOnOperation() override;
  Option<uint> numIterations{
      *this, "numIterations",
      llvm::cl::desc(
          "Number of reset attempts to break reset ops into, default is 1"),
      llvm::cl::value_desc("num"), llvm::cl::init(1)};
  Option<uint> delayCycles{
      *this, "delayCycles",
      llvm::cl::desc("Number of cycles of delay to add between reset "
                     "iterations, default is 1000"),
      llvm::cl::value_desc("num"), llvm::cl::init(1000)};

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct BreakResetPass
} // namespace mlir::quir

#endif // QUIR_BREAK_RESET_H

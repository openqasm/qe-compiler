//===- BreakReset.h - Breset reset ops --------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for breaking resets into control-flow.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_BREAK_RESET_H
#define QUIR_BREAK_RESET_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// This pass converts input ResetQubitOps to a parameterized number of
/// measure and the conditionally called x gates, as well as an optional delay.
struct BreakResetPass
    : public mlir::PassWrapper<BreakResetPass, mlir::OperationPass<ModuleOp>> {
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

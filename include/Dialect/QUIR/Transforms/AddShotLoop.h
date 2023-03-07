//===- AddShotLoop.h - Add shot loop ----------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
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
//
//  This file declares the pass for adding a for shot loop around the entire
//  main function body
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_ADD_SHOT_LOOP_H
#define QUIR_ADD_SHOT_LOOP_H

#include "Dialect/QUIR/IR/QUIRDialect.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir::quir {
struct AddShotLoopPass
    : public PassWrapper<AddShotLoopPass, OperationPass<ModuleOp>> {
  AddShotLoopPass() = default;
  AddShotLoopPass(const AddShotLoopPass &pass) : PassWrapper(pass) {}
  AddShotLoopPass(uint inNumShots, uint inShotDelayCycles) {
    numShots = inNumShots;
    shotDelayCycles = inShotDelayCycles;
  }

  void runOnOperation() override;

  Option<uint> numShots{*this, "num-shots",
                        llvm::cl::desc("Number of shots, default is 1000"),
                        llvm::cl::value_desc("num"), llvm::cl::init(1000)};
  Option<uint> shotDelayCycles{
      *this, "shot-delay",
      llvm::cl::desc("Cycles of delay (dt) to insert between shots, default is "
                     "4499200(1ms repetition delay at 4.5GS/s)"),
      llvm::cl::value_desc("num"), llvm::cl::init(4499200)};

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::quir::QUIRDialect>();
  }
}; // struct AddShotLoopPass
} // namespace mlir::quir

#endif // QUIR_ADD_SHOT_LOOP_H

//===- QUIRToStd.h - Convert QUIR to Std Dialect ----------------*- C++ -*-===//
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
//
//  This file declares the pass for converting QUIR to std dialect
//
//===----------------------------------------------------------------------===//

#ifndef SIMULATOR_CONVERSION_QUIRTOSTD_H
#define SIMULATOR_CONVERSION_QUIRTOSTD_H

#include "Simulator.h"

#include "HAL/TargetOperationPass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace qssc::targets::simulator::conversion {
struct SimulatorQUIRToStdPass
    : public mlir::PassWrapper<
          SimulatorQUIRToStdPass,
          hal::TargetOperationPass<SimulatorSystem, mlir::ModuleOp>> {
  void runOnOperation(SimulatorSystem &system) override;
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  bool externalizeOutputVariables;

  SimulatorQUIRToStdPass(bool externalizeOutputVariables)
      : PassWrapper(), externalizeOutputVariables(externalizeOutputVariables) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};
} // namespace qssc::targets::simulator::conversion

#endif // SIMULATOR_CONVERSION_QUIRTOSTD_H

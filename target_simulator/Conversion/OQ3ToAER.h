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
//  This file declares the pass for converting OpenQASM3 to AER
//
//===----------------------------------------------------------------------===//

#ifndef SIMULATOR_CONVERSION_OQ3AER_H
#define SIMULATOR_CONVERSION_OQ3AER_H

#include "Simulator.h"

#include "HAL/TargetOperationPass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace qssc::targets::simulator::conversion {
struct QUIRToAERPass
    : public mlir::PassWrapper<
          QUIRToAERPass,
          hal::TargetOperationPass<SimulatorSystem, mlir::ModuleOp>> {
  void runOnOperation(SimulatorSystem &system) override;
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  
  bool externalizeOutputVariables;

  QUIRToAERPass(bool externalizeOutputVariables)
      : PassWrapper(), externalizeOutputVariables(externalizeOutputVariables)
  {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  
private:
  void declareAerFunctions(mlir::ModuleOp moduleOp);
};
} // namespace qssc::targets::simulator::conversion

#endif // SIMULATOR_CONVERSION_OQ3AER_H

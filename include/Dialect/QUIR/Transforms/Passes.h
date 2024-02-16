//===- Passes.h - Quir Passes -----------------------------------*- C++ -*-===//
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

#ifndef QUIR_QUIRPASSES_H
#define QUIR_QUIRPASSES_H

#include "AddShotLoop.h"
#include "AngleConversion.h"
#include "BreakReset.h"
#include "ConvertDurationUnits.h"
#include "FunctionArgumentSpecialization.h"
#include "LoadElimination.h"
#include "MergeCircuitMeasures.h"
#include "MergeCircuits.h"
#include "MergeMeasures.h"
#include "MergeParallelResets.h"
#include "QuantumDecoration.h"
#include "RemoveQubitOperands.h"
#include "RemoveUnusedCircuits.h"
#include "ReorderCircuits.h"
#include "ReorderMeasurements.h"
#include "SubroutineCloning.h"
#include "UnusedVariable.h"
#include "VariableElimination.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::quir {
void registerQuirPasses();
void registerQuirPassPipeline();

// This pass recurses through the IR and detects when scf
// ops use only classical operations. It then applies the classicalOnly
// attribute to all of the scf ops with a value of either true or false
struct ClassicalOnlyDetectionPass
    : public PassWrapper<ClassicalOnlyDetectionPass, OperationPass<>> {
  auto hasQuantumSubOps(Operation *inOp) -> bool;
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
}; // end struct ClassicalOnlyDetectionPass

} // namespace mlir::quir

#endif // QUIR_QUIRPASSES_H

//===- Passes.h - Quir Passes -----------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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

#ifndef QUIR_QUIRPASSES_H
#define QUIR_QUIRPASSES_H

#include "AddShotLoop.h"
#include "AngleConversion.h"
#include "BreakReset.h"
#include "FunctionArgumentSpecialization.h"
#include "LoadElimination.h"
#include "MergeMeasures.h"
#include "MergeParallelResets.h"
#include "QuantumDecoration.h"
#include "RemoveQubitOperands.h"
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
}; // end struct ClassicalOnlyDetectionPass

} // namespace mlir::quir

#endif // QUIR_QUIRPASSES_H

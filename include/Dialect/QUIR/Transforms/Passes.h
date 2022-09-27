//===- Passes.h - Quir Passes -----------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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
#include "FunctionArgumentSpecialization.h"
#include "LoadElimination.h"
#include "MergeMeasures.h"
#include "MergeParallelResets.h"
#include "QuantumDecoration.h"
#include "RemoveQubitArgs.h"
#include "SubroutineCloning.h"
#include "UnusedVariable.h"

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

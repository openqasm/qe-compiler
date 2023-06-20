//===- ParameterInitialValueAnalysis.cpp - initial_value cache ---- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file defines a MLIR Analysis for parameter inputs which
/// caches the initial_value of the input parameter
///
/// Note: by default this analysis is always treated as valid unless
/// the invalidate() method is called.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/Utils/ParameterInitialValueAnalysis.h"

#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "ParameterInitialValueAnalysis"

using namespace mlir::qcs;

ParameterInitialValueAnalysis::ParameterInitialValueAnalysis(
    mlir::Operation *op) {
  op->walk([&](DeclareParameterOp declareParameterOp) {
    double initial_value = 0.0;
    if (declareParameterOp.initial_value().hasValue()) {
      auto angleAttr = declareParameterOp.initial_value()
                           .getValue()
                           .dyn_cast<mlir::quir::AngleAttr>();
      auto floatAttr =
          declareParameterOp.initial_value().getValue().dyn_cast<FloatAttr>();
      if (!(angleAttr || floatAttr))
        op->emitError(
            "Parameters are currently limited to angles or float[64] only.");

      if (angleAttr)
        initial_value = angleAttr.getValue().convertToDouble();

      if (floatAttr)
        initial_value = floatAttr.getValue().convertToDouble();
    }
    initial_values_[declareParameterOp.sym_name().str()] = initial_value;
  });
  invalid_ = false;
}

void ParameterInitialValueAnalysisPass::runOnOperation() {
  getAnalysis<ParameterInitialValueAnalysis>();
} // ParameterInitialValueAnalysisPass::runOnOperation()

llvm::StringRef ParameterInitialValueAnalysisPass::getArgument() const {
  return "qcs-parameter-initial-value-analysis";
}

llvm::StringRef ParameterInitialValueAnalysisPass::getDescription() const {
  return "Run ParameterIntialValueAnalysis";
}

// TODO: move registerQCSPasses to separate source file if additional passes
// are added to the QCS Dialect
void mlir::qcs::registerQCSPasses() {
  PassRegistration<mlir::qcs::ParameterInitialValueAnalysisPass>();
}

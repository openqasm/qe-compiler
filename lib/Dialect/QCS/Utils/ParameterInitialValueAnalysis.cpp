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

#define DEBUG_TYPE "ParameterInitalValueAnalysis"

using namespace mlir::qcs;

ParameterInitalValueAnalysis::ParameterInitalValueAnalysis(
    mlir::Operation *op) {
  op->walk([&](DeclareParameterOp declareParameterOp) {
    double initial_value = 0.0;
    if (declareParameterOp.initial_value().hasValue()) {
      auto angleAttr = declareParameterOp.initial_value()
                           .getValue()
                           .dyn_cast<mlir::quir::AngleAttr>();
      if (!angleAttr)
        op->emitError("Parameters are currently limited to angles only.");
      else
        initial_value = angleAttr.getValue().convertToDouble();
    }
    initial_values_[declareParameterOp.sym_name().str()] = initial_value;
  });
  invalid_ = false;
}

void ParameterInitalValueAnalysisPass::runOnOperation() {
  getAnalysis<ParameterInitalValueAnalysis>();
} // ParameterInitalValueAnalysisPass::runOnOperation()

llvm::StringRef ParameterInitalValueAnalysisPass::getArgument() const {
  return "oq2-parameter-names-analysis";
}

llvm::StringRef ParameterInitalValueAnalysisPass::getDescription() const {
  return "Run ParameterNamesAnalysis";
}
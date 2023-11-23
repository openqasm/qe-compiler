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
    mlir::Operation *moduleOp) {

  if (not invalid_)
    return;

  // process the module top level to cache declareParameterOp initial_values
  // this does not use a walk method so that submodule (if present) are not
  // processed in order to limit processing time

  for (auto &region : moduleOp->getRegions())
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        auto declareParameterOp = dyn_cast<DeclareParameterOp>(op);
        if (!declareParameterOp)
          continue;

        // moduleOp->walk([&](DeclareParameterOp declareParameterOp) {
        double initial_value = 0.0;
        if (declareParameterOp.getInitialValue().has_value()) {
          auto angleAttr = declareParameterOp.getInitialValue()
                               .value()
                               .dyn_cast<mlir::quir::AngleAttr>();
          auto floatAttr = declareParameterOp.getInitialValue()
                               .value()
                               .dyn_cast<FloatAttr>();
          if (!(angleAttr || floatAttr))
            declareParameterOp.emitError("Parameters are currently limited to "
                                         "angles or float[64] only.");

          if (angleAttr)
            initial_value = angleAttr.getValue().convertToDouble();

          if (floatAttr)
            initial_value = floatAttr.getValue().convertToDouble();
        }
        initial_values_[declareParameterOp.getSymName()] = initial_value;
      }
  invalid_ = false;
}

void ParameterInitialValueAnalysisPass::runOnOperation() {
  getAnalysis<ParameterInitialValueAnalysis>();
} // ParameterInitialValueAnalysisPass::runOnOperation()

llvm::StringRef ParameterInitialValueAnalysisPass::getArgument() const {
  return "qcs-parameter-initial-value-analysis";
}

llvm::StringRef ParameterInitialValueAnalysisPass::getDescription() const {
  return "Run ParameterInitialValueAnalysis";
}

// TODO: move registerQCSPasses to separate source file if additional passes
// are added to the QCS Dialect
void mlir::qcs::registerQCSPasses() {
  PassRegistration<mlir::qcs::ParameterInitialValueAnalysisPass>();
}

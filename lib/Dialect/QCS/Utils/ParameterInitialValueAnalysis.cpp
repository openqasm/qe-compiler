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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ParameterInitialValueAnalysis"

using namespace mlir::qcs;

static llvm::cl::opt<bool> printAnalysisEntries(
    "qcs-parameter-initial-value-analysis-print",
    llvm::cl::desc("Print ParameterInitialValueAnalysis entries"),
    llvm::cl::init(false));

ParameterInitialValueAnalysis::ParameterInitialValueAnalysis(
    mlir::Operation *moduleOp) {

  if (not invalid_)
    return;

  bool foundParameters = false;

  // search for the parameters in the current module
  // if not found search parent module
  // TODO - determine if there is a faster way to do this
  do {

    // process the module top level to cache declareParameterOp initial_values
    // this does not use a walk method so that submodule (if present) are not
    // processed in order to limit processing time

    for (auto &region : moduleOp->getRegions())
      for (auto &block : region.getBlocks())
        for (auto &op : block.getOperations()) {
          auto parameterLoadOp = dyn_cast<ParameterLoadOp>(op);
          if (!parameterLoadOp)
            continue;

          double initialValue =
              std::get<double>(parameterLoadOp.getInitialValue());
          initialValues_[parameterLoadOp.getParameterName()] = initialValue;
          foundParameters = true;
        }
    if (!foundParameters) {
      auto parentOp = moduleOp->getParentOfType<mlir::ModuleOp>();
      if (parentOp)
        moduleOp = parentOp;
      else
        break;
    }
  } while (!foundParameters);
  invalid_ = false;

  // debugging / test print out
  if (printAnalysisEntries) {
    for (auto &initialValue : initialValues_) {
      llvm::outs() << initialValue.first() << " = "
                   << std::get<double>(initialValue.second) << "\n";
    }
  }
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

llvm::StringRef ParameterInitialValueAnalysisPass::getName() const {
  return "Parameters Initial Value Analysis Pass";
}

// TODO: move registerQCSPasses to separate source file if additional passes
// are added to the QCS Dialect
void mlir::qcs::registerQCSPasses() {
  PassRegistration<mlir::qcs::ParameterInitialValueAnalysisPass>();
}

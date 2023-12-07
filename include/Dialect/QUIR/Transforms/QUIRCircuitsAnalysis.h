//===- QUIRCircuitsAnalysis.h - Cache circuit argument values ---*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file implements an analysis for caching argument attributes with
/// default values for angle and duration arguments.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_CIRCUITS_ANALYSIS_H
#define QUIR_CIRCUITS_ANALYSIS_H

#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/QCS/Utils/ParameterInitialValueAnalysis.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"

#include <tuple>

namespace mlir::quir {

using OperandAttributes =
    std::tuple<double, llvm::StringRef, mlir::quir::DurationAttr>;

using CircuitAnalysisMap = std::unordered_map<
    mlir::Operation *,
    std::unordered_map<mlir::Operation *,
                       std::unordered_map<int, OperandAttributes>>>;

class QUIRCircuitsAnalysis {
private:
  CircuitAnalysisMap circuitOperands;
  bool invalid_{true};

public:
  QUIRCircuitsAnalysis(mlir::Operation *op, AnalysisManager &am);
  CircuitAnalysisMap &getAnalysisMap() { return circuitOperands; }

  void invalidate() { invalid_ = true; }
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return invalid_;
  }
};

struct QUIRCircuitsAnalysisPass
    : public mlir::PassWrapper<QUIRCircuitsAnalysisPass,
                               OperationPass<ModuleOp>> {

  QUIRCircuitsAnalysisPass() = default;
  QUIRCircuitsAnalysisPass(const QUIRCircuitsAnalysisPass &pass) = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // QUIRCircuitsAnalysisPass

llvm::Expected<double>
angleValToDouble(mlir::Value inVal,
                 mlir::qcs::ParameterInitialValueAnalysis *nameAnalysis,
                 mlir::quir::QUIRCircuitsAnalysis *circuitAnalysis = nullptr);

} // namespace mlir::quir

#endif // QUIR_CIRCUITS_ANALYSIS_H

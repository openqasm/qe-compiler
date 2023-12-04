//===- LabelQUIRCircuits.h - Add attrs for circuit arguments  ---*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file implements a pass for adding argument attributes with default
/// values for angle and duration arguments.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_CIRCUITS_ANALYSIS_H
#define QUIR_CIRCUITS_ANALYSIS_H

#include "Dialect/Pulse/IR/PulseOps.h"
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
  // void dump() {
  //   for (auto entry : circuitOperands) {
  //     auto circuitOp = dyn_cast<quir::CircuitOp>(entry.first);
  //     llvm::errs() << "circuit: " << circuitOp.sym_name() << "\n";
  //     for (auto arg : entry.second) {
  //       llvm::errs() << "arg: " << arg.first << " ";
  //       std::get<2>(arg.second).dump();
  //   }
  //   }
  // }

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

} // namespace mlir::quir

#endif // QUIR_CIRCUITS_ANALYSIS_H

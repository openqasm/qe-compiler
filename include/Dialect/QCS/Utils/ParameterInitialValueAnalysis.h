//===- ParameterInitialValueAnalysis.h - initial_value cache ------ C++ -*-===//
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

#ifndef QCS_PARAMETER_INITIAL_VALUE_ANALYSIS_H
#define QCS_PARAMETER_INITIAL_VALUE_ANALYSIS_H

#include "Dialect/QCS/IR/QCSOps.h"
#include "HAL/SystemConfiguration.h"

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <unordered_map>

namespace mlir::qcs {

using namespace mlir;

using InitialValueType = llvm::StringMap<ParameterType>;

class ParameterInitialValueAnalysis {
private:
  InitialValueType initial_values_;
  bool invalid_{true};

public:
  ParameterInitialValueAnalysis(mlir::Operation *op);
  InitialValueType &getNames() { return initial_values_; }
  void invalidate() { invalid_ = true; }
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return invalid_;
  }
};

struct ParameterInitialValueAnalysisPass
    : public PassWrapper<ParameterInitialValueAnalysisPass, OperationPass<>> {

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName =
      "Parameter Initial Value Analysis Pass (" + getArgument().str() + ")";
}; // struct ParameterInitialValueAnalysisPass

// TODO: move registerQCSPasses to separate header if additional passes
// are added to the QCS Dialect
void registerQCSPasses();

} // namespace mlir::qcs

#endif // QCS_PARAMETER_INITIAL_VALUE_ANALYSIS_H

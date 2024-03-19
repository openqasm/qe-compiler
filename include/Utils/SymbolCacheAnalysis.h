//===- SymbolCacheAnalysis.h - Cache symbols --------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file implements an analysis for caching symbols for quir.circuits and
/// pulse.sequence
///
//===----------------------------------------------------------------------===//

#ifndef CACHE_SYMBOLS_ANALYSIS_H
#define CACHE_SYMBOLS_ANALYSIS_H

#include "HAL/SystemConfiguration.h"

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Operation.h>

namespace qssc::utils {

class SymbolCacheAnalysis {
private:
  llvm::StringMap<mlir::Operation *> symbolOpsMap;
  std::unordered_map<mlir::Operation *, mlir::Operation *> callMap;
  mlir::Operation *topOp{nullptr};
  bool invalid_{true};

public:
  SymbolCacheAnalysis(mlir::Operation *op) {
    llvm::errs() << "SymbolCacheAnalysis" << "\n";
    topOp = op;
  }
  SymbolCacheAnalysis(mlir::Operation *op, qssc::hal::SystemConfiguration *config) {
    llvm::errs() << "SymbolCacheAnalysis with config" << "\n";
    topOp = op;
  }
  llvm::StringMap<mlir::Operation *> &getSymbolMap() { return symbolOpsMap; }
  std::unordered_map<mlir::Operation *, mlir::Operation *>  &getCallMap() { return callMap; }

  template<class CalleeOp>
  SymbolCacheAnalysis &addToCache() {
    return addToCache<CalleeOp>(topOp);
  }

  template<class CalleeOp>
  SymbolCacheAnalysis &addToCache(mlir::Operation *op) {
    op->walk([&](CalleeOp op) {
        symbolOpsMap[op.getSymName()] = op.getOperation();
    });
    return *this;
  }

  template<class CallOp>
  SymbolCacheAnalysis cacheCallMap() {
     return cacheCallMap<CallOp>(topOp);
  }

  template<class CallOp>
  SymbolCacheAnalysis cacheCallMap(mlir::Operation *op) {
    op->walk([&](CallOp callOp) {
        auto search = symbolOpsMap.find(callOp.getCallee());
        if (search != symbolOpsMap.end())
          callMap[callOp.getOperation()] = search->second;
    });
    return *this;
  }

  template<class CalleeOp>
  CalleeOp getOpByName(llvm::StringRef callee) {
    auto search = symbolOpsMap.find(callee);
    assert(search != symbolOpsMap.end() && "matching callee not found");
    auto calleeOp = dyn_cast<CalleeOp>(search->second);
    assert(calleeOp && "callee is not of the expected type");
    return calleeOp;
  }

  template<class CalleeOp, class CallOp>
  CalleeOp getOpByCall(CallOp callOp) {
    auto search = callMap.find(callOp.getOperation());
    if (search == callMap.end()) {
      auto calleeOp =  getOpByName<CalleeOp>(callOp.getCallee());
      callMap[callOp.getOperation()] = calleeOp.getOperation();
      return calleeOp;
    }
    assert(search != callMap.end() && "matching callee not found");
    auto calleeOp = dyn_cast<CalleeOp>(search->second);
    assert(calleeOp && "callee is not of the expected type");
    return calleeOp;
  }

  template<class CalleeOp, class CallOp>
  CalleeOp getOp(CallOp callOp) {
    return getOpByCall<CalleeOp, CallOp>(callOp);
  }

  template<class CalleeOp>
  void addCallee(CalleeOp calleeOp) {
    symbolOpsMap[calleeOp.getSymName()] = calleeOp.getOperation();
  }

  template<class CallOp, class CalleeOp>
  void cacheCall(CallOp callOp, CalleeOp calleeOp) {
    callMap[callOp.getOperation()] = calleeOp.getOperation();
  }

  void invalidate() { 
    symbolOpsMap.clear();
    callMap.clear();
    invalid_ = true; 
  }

  void freeze() {
    invalid_ = false;
  }

  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return invalid_;
  }

  // for debugging purposes
  void listSymbols() {
    for (auto & [key, value] : symbolOpsMap)
      llvm::errs() << key << "\n";
  }
};
} // namespace qssc::utils

#endif // CACHE_SYMBOLS_ANALYSIS_H

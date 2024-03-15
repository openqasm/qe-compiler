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
  mlir::Operation *topOp{nullptr};

public:
  SymbolCacheAnalysis(mlir::Operation *op) {
    topOp = op;
  }
  SymbolCacheAnalysis(mlir::Operation *op, qssc::hal::SystemConfiguration *config) {
    topOp = op;
  }
  llvm::StringMap<mlir::Operation *> &getSymbolMap() { return symbolOpsMap; }

  template<class CalleeOp>
  SymbolCacheAnalysis &addToCache() {
    topOp->walk([&](CalleeOp op) {
        symbolOpsMap[op.getSymName()] = op.getOperation();
    });
    return *this;
  }

  template<class CalleeOp>
  CalleeOp getOp(llvm::StringRef callee) {
    auto search = symbolOpsMap.find(callee);
    llvm::errs() << callee << "\n";
    assert(search != symbolOpsMap.end() && "matching callee not found");
    auto calleeOp = dyn_cast<CalleeOp>(search->second);
    assert(calleeOp && "callee is not of the expected type");
    return calleeOp;
  }


  // for debugging purposes
  void listSymbols() {
    for (auto & [key, value] : symbolOpsMap)
      llvm::errs() << key << "\n";
  }
};
} // namespace qssc::utils

#endif // CACHE_SYMBOLS_ANALYSIS_H

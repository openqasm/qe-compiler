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
/// This file implements an analysis for caching symbols for which match a
/// call -> callee pattern. This currently includes circuit / call_circuit
/// and sequence / call_sequence.
///
///
//===----------------------------------------------------------------------===//

#ifndef CACHE_SYMBOLS_ANALYSIS_H
#define CACHE_SYMBOLS_ANALYSIS_H

#include "HAL/SystemConfiguration.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

namespace qssc::utils {

// This analysis maintains a mapping of symbol name to operation in
// symbolOpsMap. It will also maintain a cache of CallOp to CalleeOp
// when operations are looked up through the getOp method. The CallOp
// to CalleeOp cache is intended to reduce string comparison where
// possible
//
// Example usage:
// auto & cache = getAnalysis<qssc::utils::SymbolCacheAnalysis>()
//                .addToCache<CircuitOp>();
//
// multiple symbol types may be cached using:
// auto & cache = getAnalysis<qssc::utils::SymbolCacheAnalysis>()
//                .addToCache<CircuitOp>()
//                .addToCache<SequenceOp>();
//
// This analysis is intended to be used with MLIR's getAnalysis
// framework. It has been designed to reused the chached value
// and will not be invalidated automatically with each pass.
// If a pass manipulates the symbols that are cached with this
// analysis then it should use the addCallee method to update the
// map or call invalidate after appying updates.
// Note this analysis should always be used by reference or
// via a pointer to ensure that updates are applied to the maps
// stored by the MLIR analysis framework.
//
// Passes may force the maps to be re-loaded by calling invalidate
// before calling addToCache:
//
// auto & cache = getAnalysis<qssc::utils::SymbolCacheAnalysis>()
//                .invalidate()
//                .addToCache<CircuitOp>();

class SymbolCacheAnalysis {
public:
  SymbolCacheAnalysis(mlir::Operation *op) {
    if (topOp && topOp != op)
      invalidate();
    topOp = op;
  }
  SymbolCacheAnalysis(mlir::Operation *op,
                      qssc::hal::SystemConfiguration *config) {
    if (topOp && topOp != op)
      invalidate();
    topOp = op;
  }

  template <class CalleeOp>
  SymbolCacheAnalysis &addToCache() {
    return addToCache<CalleeOp>(topOp);
  }

  template <class CalleeOp>
  SymbolCacheAnalysis &addToCache(mlir::Operation *op) {
    std::string typeName = typeid(CalleeOp).name();

    if (!invalid && (cachedTypes.find(typeName) != cachedTypes.end())) {
      // already cached skipping
      return *this;
    }

    op->walk([&](CalleeOp op) {
      symbolOpsMap[op.getSymName()] = op.getOperation();
    });
    cachedTypes.insert(typeName);
    invalid = false;
    return *this;
  }

  template <class CallOp>
  SymbolCacheAnalysis &cacheCallMap() {
    return cacheCallMap<CallOp>(topOp);
  }

  template <class CallOp>
  SymbolCacheAnalysis &cacheCallMap(mlir::Operation *op) {
    std::string typeName = typeid(CallOp).name();
    if ((cachedTypes.find(typeName) != cachedTypes.end()))
      return *this;

    op->walk([&](CallOp callOp) {
      auto search = symbolOpsMap.find(callOp.getCallee());
      if (search != symbolOpsMap.end())
        callMap[callOp.getOperation()] = search->second;
    });
    return *this;
  }

  template <class CalleeOp>
  CalleeOp getOpByName(llvm::StringRef callee) {
    auto search = symbolOpsMap.find(callee);
    assert(search != symbolOpsMap.end() && "matching callee not found");
    auto calleeOp = llvm::dyn_cast<CalleeOp>(search->second);
    assert(calleeOp && "callee is not of the expected type");
    return calleeOp;
  }

  template <class CalleeOp, class CallOp>
  CalleeOp getOpByCall(CallOp callOp) {
    auto search = callMap.find(callOp.getOperation());
    if (search == callMap.end()) {
      auto calleeOp = getOpByName<CalleeOp>(callOp.getCallee());
      callMap[callOp.getOperation()] = calleeOp.getOperation();
      return calleeOp;
    }
    auto calleeOp = llvm::dyn_cast<CalleeOp>(search->second);
    assert(calleeOp && "callee is not of the expected type");
    return calleeOp;
  }

  template <class CalleeOp, class CallOp>
  CalleeOp getOp(CallOp callOp) {
    return getOpByCall<CalleeOp, CallOp>(callOp);
  }

  template <class CalleeOp>
  void addCallee(CalleeOp calleeOp) {
    addCallee(calleeOp.getSymName(), calleeOp.getOperation());
  }

  void addCallee(llvm::StringRef name, mlir::Operation *op) {
    // if this is an update to existing symbol clear callMap cache
    if (symbolOpsMap.contains(name))
      callMap.clear();
    symbolOpsMap[name] = op;
  }

  template <class CallOp, class CalleeOp>
  void cacheCall(CallOp callOp, CalleeOp calleeOp) {
    callMap[callOp.getOperation()] = calleeOp.getOperation();
  }

  bool contains(llvm::StringRef name) { return symbolOpsMap.contains(name); }

  template <class CalleeOp>
  void erase(CalleeOp calleeOp) {
    symbolOpsMap.erase(calleeOp.getSymName());
    // TODO: determine if it is worth just clearing the callers of calleeOp
    callMap.clear();
  }

  SymbolCacheAnalysis &invalidate() {
    symbolOpsMap.clear();
    callMap.clear();
    cachedTypes.clear();
    invalid = true;
    return *this;
  }

  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return invalid;
  }

  // for debugging purposes
  void listSymbols() {
    for (auto &[key, value] : symbolOpsMap)
      llvm::errs() << key << "\n";
  }

private:
  llvm::StringMap<mlir::Operation *> symbolOpsMap;
  std::unordered_map<mlir::Operation *, mlir::Operation *> callMap;
  std::unordered_set<std::string> cachedTypes;
  mlir::Operation *topOp{nullptr};
  bool invalid{true};
};
} // namespace qssc::utils

#endif // CACHE_SYMBOLS_ANALYSIS_H

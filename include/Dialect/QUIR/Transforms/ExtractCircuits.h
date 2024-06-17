//===- ExtractCircuits.h - Extract circuits ops -----------------*- C++ -*-===//
//
// (C) Copyright IBM 2024.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for extracting quantum ops into quir.circuits
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_EXTRACT_CIRCUITS_H
#define QUIR_EXTRACT_CIRCUITS_H

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Utils/SymbolCacheAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <set>
#include <unordered_map>

namespace mlir::quir {

struct ExtractCircuitsPass
    : public PassWrapper<ExtractCircuitsPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;

private:
  void processRegion(mlir::Region &region, OpBuilder topLevelBuilder,
                     OpBuilder circuitBuilder);
  void processBlock(mlir::Block &block, OpBuilder topLevelBuilder,
                    OpBuilder circuitBuilder);
  OpBuilder startCircuit(mlir::Location location, OpBuilder topLevelBuilder);
  void endCircuit(mlir::Operation *firstOp, mlir::Operation *lastOp,
                  OpBuilder topLevelBuilder, OpBuilder circuitBuilder);
  void addToCircuit(mlir::Operation *currentOp, OpBuilder circuitBuilder);
  uint64_t circuitCount = 0;
  qssc::utils::SymbolCacheAnalysis *symbolCache{nullptr};

  mlir::quir::CircuitOp currentCircuitOp = nullptr;
  mlir::IRMapping currentCircuitMapper;
  mlir::quir::CallCircuitOp newCallCircuitOp;

  llvm::SmallVector<Type> inputTypes;
  llvm::SmallVector<Value> inputValues;
  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;
  std::vector<int> phyiscalIds;
  std::unordered_map<uint32_t, int> argToId;

  std::unordered_map<Operation *, uint32_t> circuitOperands;
  llvm::SmallVector<OpResult> originalResults;
  std::set<Operation *> eraseSet;

}; // struct ExtractCircuitsPass
} // namespace mlir::quir
#endif // QUIR_EXTRACT_CIRCUITS_H

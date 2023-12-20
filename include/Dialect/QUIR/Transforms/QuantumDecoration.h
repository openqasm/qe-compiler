//===- QuantumDecoration.h - Add quantum attributes -------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
/// This file declares the pass for decorating ops that have regions (if, for,
/// etc.) with attributes describing the number and id of qubits inside it
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUANTUM_DECORATION_H
#define QUIR_QUANTUM_DECORATION_H

#include "Dialect/QUIR/IR/QUIROps.h"
#include "mlir/Pass/Pass.h"

#include <unordered_set>

namespace mlir {
class ModuleOp;
} // end namespace mlir

namespace mlir::quir {
struct QuantumDecorationPass
    : public PassWrapper<QuantumDecorationPass, OperationPass<ModuleOp>> {
  // TODO: Add a mechanism to get the qubit arguments for any qubit-using op
  // using a standard interface, so this can be simplified to a single function
  void processOp(Operation *op, std::unordered_set<int> &retSet);
  void processOp(BuiltinCXOp op, std::unordered_set<int> &retSet);
  void processOp(Builtin_UOp op, std::unordered_set<int> &retSet);
  void processOp(CallDefCalGateOp op, std::unordered_set<int> &retSet);
  void processOp(CallDefcalMeasureOp op, std::unordered_set<int> &retSet);
  void processOp(DelayOp op, std::unordered_set<int> &retSet);
  void processOp(CallGateOp op, std::unordered_set<int> &retSet);
  void processOp(BarrierOp op, std::unordered_set<int> &retSet);
  void processOp(MeasureOp op, std::unordered_set<int> &retSet);
  void processOp(ResetQubitOp op, std::unordered_set<int> &retSet);
  void processOp(CallCircuitOp op, std::unordered_set<int> &retSet);
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName =
      "Quantum Decoration Pass (" + getArgument().str() + ")";
}; // QuantumDecorationPass
} // namespace mlir::quir

#endif // QUIR_QUANTUM_DECORATION_H

//===- QuantumDecoration.h - Add quantum attributes -------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // QuantumDecorationPass
} // namespace mlir::quir

#endif // QUIR_QUANTUM_DECORATION_H

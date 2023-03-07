//===- QuantumDecoration.cpp - Add quantum attributes -----------*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
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
/// This file implements the pass for decorating ops that have regions (if, for,
/// etc.) with attributes describing the number and id of qubits inside it
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/QuantumDecoration.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

namespace {
int lookupOrMinus1(Value arg) {
  auto id = lookupQubitId(arg);
  if (id)
    return static_cast<int>(*id);
  return -1;
}
} // end anonymous namespace

void QuantumDecorationPass::processOp(Operation *op,
                                      std::unordered_set<int> &retSet) {
  if (auto castOp = dyn_cast<BuiltinCXOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<Builtin_UOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<CallDefCalGateOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<CallGateOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<BarrierOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<MeasureOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<CallDefcalMeasureOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<DelayOp>(op))
    processOp(castOp, retSet);
  else if (auto castOp = dyn_cast<ResetQubitOp>(op))
    processOp(castOp, retSet);
} // processOp Operation *

void QuantumDecorationPass::processOp(Builtin_UOp builtinUOp,
                                      std::unordered_set<int> &retSet) {
  retSet.emplace(lookupOrMinus1(builtinUOp.target()));
} // processOp Builtin_UOp

void QuantumDecorationPass::processOp(BuiltinCXOp builtinCXOp,
                                      std::unordered_set<int> &retSet) {
  retSet.emplace(lookupOrMinus1(builtinCXOp.control()));
  retSet.emplace(lookupOrMinus1(builtinCXOp.target()));
} // processOp BuiltinCXOp

void QuantumDecorationPass::processOp(MeasureOp measureOp,
                                      std::unordered_set<int> &retSet) {
  for (auto qubit : measureOp.qubits())
    retSet.emplace(lookupOrMinus1(qubit));
} // processOp MeasureOp

void QuantumDecorationPass::processOp(CallDefcalMeasureOp measureOp,
                                      std::unordered_set<int> &retSet) {
  std::vector<Value> qubitOperands;
  qubitCallOperands(measureOp, qubitOperands);

  for (Value &val : qubitOperands)
    retSet.emplace(lookupOrMinus1(val));
} // processOp MeasureOp

void QuantumDecorationPass::processOp(DelayOp delayOp,
                                      std::unordered_set<int> &retSet) {
  for (auto qubit_operand : delayOp.qubits())
    retSet.emplace(lookupOrMinus1(qubit_operand));
} // processOp MeasureOp

void QuantumDecorationPass::processOp(ResetQubitOp resetOp,
                                      std::unordered_set<int> &retSet) {
  for (auto qubit : resetOp.qubits())
    retSet.emplace(lookupOrMinus1(qubit));
} // processOp MeasureOp

void QuantumDecorationPass::processOp(CallDefCalGateOp callOp,
                                      std::unordered_set<int> &retSet) {
  std::vector<Value> qubitOperands;
  qubitCallOperands(callOp, qubitOperands);

  for (Value &val : qubitOperands)
    retSet.emplace(lookupOrMinus1(val));
} // processOp CallGateOp

void QuantumDecorationPass::processOp(CallGateOp callOp,
                                      std::unordered_set<int> &retSet) {
  std::vector<Value> qubitOperands;
  qubitCallOperands(callOp, qubitOperands);

  for (Value &val : qubitOperands)
    retSet.emplace(lookupOrMinus1(val));
} // processOp CallGateOp

void QuantumDecorationPass::processOp(BarrierOp barrierOp,
                                      std::unordered_set<int> &retSet) {
  std::vector<Value> qubitOperands;
  qubitCallOperands(barrierOp, qubitOperands);

  for (Value &val : qubitOperands)
    retSet.emplace(lookupOrMinus1(val));
} // processOp BarrierOp

void QuantumDecorationPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  OpBuilder build(moduleOp);

  moduleOp->walk([&](Operation *op) {
    if (dyn_cast<scf::IfOp>(op) || dyn_cast<scf::ForOp>(op) ||
        dyn_cast<quir::SwitchOp>(op)) {
      std::unordered_set<int> involvedQubits;
      op->walk([&](Operation *subOp) { processOp(subOp, involvedQubits); });
      std::vector<int> qubitVec;
      qubitVec.reserve(involvedQubits.size());
      qubitVec.insert(qubitVec.begin(), involvedQubits.begin(),
                      involvedQubits.end());
      std::sort(qubitVec.begin(), qubitVec.end());
      op->setAttr(mlir::quir::getPhysicalIdsAttrName(),
                  build.getI32ArrayAttr(ArrayRef<int>(qubitVec)));
    }
  });

} // runOnOperation

llvm::StringRef QuantumDecorationPass::getArgument() const {
  return "quantum-decorate";
}
llvm::StringRef QuantumDecorationPass::getDescription() const {
  return "Detect and add attributes to ops describing which "
         "qubits are involved within those ops";
}

//===- Utils.h - QUIR Utilities ---------------------------------*- C++ -*-===//
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
//
//  This file declares some utility functions for QUIR passes
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUIRUTILS_H
#define QUIR_QUIRUTILS_H

#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/BitVector.h"

#include <set>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::quir {

template <class T1>
auto quirTypeMatch(Type type1, Type type2) -> bool;

template <class T1, class T2, class... Rest>
auto quirTypeMatch(Type type1, Type type2) -> bool;

// Matches function types independent of argument width for QUIR types
// i.e. comparing angle<20> and angle<1> and angle (no <>) will return true
auto quirFunctionTypeMatch(FunctionType &ft1, FunctionType &ft2) -> bool;

// walks the module operation and searches for a func def labeled "main"
// TODO: Should be replaced by an analysis compatable struct.
Operation *getMainFunction(Operation *moduleOperation);

// takes a module Op and returns the quir.nodeType attribute string
// TODO: Should be replaced by an analysis compatable struct.
auto getNodeType(Operation *moduleOperation) -> std::string;

/// takes a module Op and returns the quir.nodeId attribute.
// TODO: Should be replaced by an analysis compatable struct.
llvm::Expected<uint32_t> getNodeId(Operation *moduleOperation);

// adds the qubit Ids on the physicalId or physicalIds attributes to theseIds
void addQubitIdsFromAttr(Operation *operation, std::vector<uint> &theseIds);
void addQubitIdsFromAttr(Operation *operation, std::set<uint> &theseIds);

// appends all of the qubit arguments for a callOp to vec
template <class CallOpTy>
void qubitCallOperands(CallOpTy &callOp, std::vector<Value> &vec);

// appends all of the classical arguments for a callOp to vec
template <class CallOpTy>
void classicalCallOperands(CallOpTy &callOp, std::vector<Value> &vec);

// returns the number of qubit arguments for a callOp
template <class CallOpTy>
void qubitCallOperands(CallOpTy &callOp, std::vector<Value> &vec) {
  for (auto arg : callOp.getOperands())
    if (arg.getType().template isa<QubitType>())
      vec.emplace_back(arg);
} // qubitCallOperands

// returns the number of classical arguments for a callOp
// TODO: Should be replaced by an analysis compatable struct.
template <class CallOpTy>
uint numClassicalCallOperands(CallOpTy &callOp);

/// appends the indices of the qubit arguments for a callOp to a bit vector
template <class CallOpTy>
void qubitArgIndices(CallOpTy &callOp, llvm::BitVector &vec);

/// Lookup a qubit id from a value.
std::optional<uint> lookupQubitId(const Value &val);

/// Get the next Op that has the CPTPOp or UnitaryOp trait, or return null if
/// none found
// TODO: Should be replaced by an analysis compatable struct.
std::optional<Operation *> nextQuantumOpOrNull(Operation *op);

/// \brief Get the next Op that has the CPTPOp or UnitaryOp trait, return it if
/// it is of type OpType, otherwise return null
// TODO: Should be replaced by an analysis compatable struct.
template <class OpType>
std::optional<OpType> nextQuantumOpOrNullOfType(Operation *op) {
  auto nextOperation = nextQuantumOpOrNull(op);
  if (nextOperation && isa<OpType>(*nextOperation))
    return dyn_cast<OpType>(*nextOperation);
  return std::nullopt;
}

/// Get the previous Op that has the CPTPOp or UnitaryOp trait, or return null
/// if none found
// TODO: Should be replaced by an analysis compatable struct.
std::optional<Operation *> prevQuantumOpOrNull(Operation *op);

/// \brief Get the previous Op that has the CPTPOp or UnitaryOp trait, return
/// it if it is of type OpType, otherwise return null
// TODO: Should be replaced by an analysis compatable struct.
template <class OpType>
std::optional<OpType> prevQuantumOpOrNullOfType(Operation *op) {
  auto prevOperation = prevQuantumOpOrNull(op);
  if (prevOperation && isa<OpType>(*prevOperation))
    return dyn_cast<OpType>(*prevOperation);
  return std::nullopt;
}

/// Get the next Op that has the CPTPOp or UnitaryOp trait, or is control flow
/// (has the RegionBranchOpInterface::Trait), or return null if none found
// TODO: Should be replaced by an analysis compatable struct.
std::optional<Operation *> nextQuantumOrControlFlowOrNull(Operation *op);

/// Get the previous Op that has the CPTPOp or UnitaryOp trait, or is control
/// flow (has the RegionBranchOpInterface::Trait), or return null if none found
// TODO: Should be replaced by an analysis compatable struct.
std::optional<Operation *> prevQuantumOrControlFlowOrNull(Operation *op);

/// \brief Check if the operation is a quantum operation
bool isQuantumOp(Operation *op);

/// Construct a DurationAttr from a ConstantOp
llvm::Expected<mlir::quir::DurationAttr>
getDuration(mlir::quir::ConstantOp &duration);
/// Construct a DurationAttr from a DelayOp
llvm::Expected<mlir::quir::DurationAttr>
getDuration(mlir::quir::DelayOp &delayOp);

// get qubit id from the result of a measurement
std::tuple<Value, MeasureOp> qubitFromMeasResult(MeasureOp measureOp,
                                                 Value result);
std::tuple<Value, MeasureOp> qubitFromMeasResult(CallCircuitOp callCircuitOp,
                                                 Value result);

} // end namespace mlir::quir

#endif // QUIR_QUIRUTILS_H

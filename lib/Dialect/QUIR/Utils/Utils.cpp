//===- Utils.cpp - QUIR Utilities -------------------------------*- C++ -*-===//
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
//  This file implements some utility functions for QUIR passes
//
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Utils/Utils.h"

#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <vector>

namespace mlir::quir {

template <class T1>
auto quirTypeMatch(Type type1, Type type2) -> bool {
  return type1.dyn_cast<T1>() && type2.dyn_cast<T1>();
}

template <class T1, class T2, class... Rest>
auto quirTypeMatch(Type type1, Type type2) -> bool {
  return (type1.dyn_cast<T1>() && type2.dyn_cast<T1>()) ||
         quirTypeMatch<T2, Rest...>(type1, type2);
}

// Matches function types independent of argument width for QUIR types
// i.e. comparing angle<20> and angle<1> and angle (no <>) will return true
auto quirFunctionTypeMatch(FunctionType &ft1, FunctionType &ft2) -> bool {
  // compare number of inputs and results
  if (ft1.getNumInputs() != ft2.getNumInputs() ||
      ft1.getNumResults() != ft2.getNumResults())
    return false;

  for (uint ii = 0; ii < ft1.getNumInputs(); ++ii) {
    Type const t1 = ft1.getInput(ii);
    Type const t2 = ft2.getInput(ii);
    if (!quirTypeMatch<AngleType, QubitType>(t1, t2) && t1 != t2)
      return false;
  }

  for (uint ii = 0; ii < ft1.getNumResults(); ++ii) {
    Type const t1 = ft1.getResult(ii);
    Type const t2 = ft2.getResult(ii);
    if (!quirTypeMatch<AngleType, QubitType>(t1, t2) && t1 != t2)
      return false;
  }

  return true;
} // quirFunctionTypeMatch

// walks the module operation and searches for a func def labeled "main"
auto getMainFunction(Operation *moduleOperation) -> Operation * {
  OpBuilder const b(moduleOperation);
  Operation *mainFunc = nullptr;
  moduleOperation->walk([&](mlir::func::FuncOp funcOp) {
    if (SymbolRefAttr::get(funcOp).getLeafReference() == "main") {
      mainFunc = funcOp.getOperation();
      return WalkResult::interrupt();
    }
    // Don't process nested values
    return WalkResult::skip();
  });
  if (!mainFunc)
    llvm::errs() << "Error: Main function not found!\n";
  return mainFunc;
} // getMainFunction

// takes a module Op and returns the quir.nodeType attribute string
auto getNodeType(Operation *moduleOperation) -> std::string {
  auto typeAttr = moduleOperation->getAttrOfType<StringAttr>("quir.nodeType");
  if (!typeAttr)
    return "";
  return typeAttr.getValue().str();
} // getNodeType

// takes a module Op and returns the quir.nodeId attribute string
llvm::Expected<uint32_t> getNodeId(Operation *moduleOperation) {
  auto typeAttr = moduleOperation->getAttrOfType<IntegerAttr>("quir.nodeId");

  if (!typeAttr)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "module Op lacks expected attribute quir.nodeId.");

  return typeAttr.getUInt();
} // getNodeType

// adds the qubit Ids on the physicalId or physicalIds attributes to theseIds
void addQubitIdsFromAttr(Operation *operation, std::vector<uint> &theseIds) {
  auto thisIdAttr = operation->getAttrOfType<IntegerAttr>(
      mlir::quir::getPhysicalIdAttrName());
  auto theseIdsAttr =
      operation->getAttrOfType<ArrayAttr>(mlir::quir::getPhysicalIdsAttrName());
  if (thisIdAttr)
    theseIds.push_back(thisIdAttr.getInt());
  if (theseIdsAttr) {
    for (Attribute const valAttr : theseIdsAttr) {
      auto intAttr = valAttr.dyn_cast<IntegerAttr>();
      theseIds.push_back(intAttr.getInt());
    }
  }
} // addQubitIdsFromAttr

// adds the qubit Ids on the physicalId or physicalIds attributes to theseIds
void addQubitIdsFromAttr(Operation *operation, std::set<uint> &theseIds) {
  auto thisIdAttr = operation->getAttrOfType<IntegerAttr>(
      mlir::quir::getPhysicalIdAttrName());
  auto theseIdsAttr =
      operation->getAttrOfType<ArrayAttr>(mlir::quir::getPhysicalIdsAttrName());
  if (thisIdAttr)
    theseIds.emplace(thisIdAttr.getInt());
  if (theseIdsAttr) {
    for (Attribute const valAttr : theseIdsAttr) {
      auto intAttr = valAttr.dyn_cast<IntegerAttr>();
      theseIds.emplace(intAttr.getInt());
    }
  }
} // addQubitIdsFromAttr

// returns a vector of all of the classical arguments for a callOp
template <class CallOpTy>
void classicalCallOperands(CallOpTy &callOp, std::vector<Value> &vec) {
  for (auto arg : callOp.getOperands())
    if (!arg.getType().template isa<QubitType>())
      vec.emplace_back(arg);
} // classicalCallOperands

// explicit template instantiation
template void classicalCallOperands<CallGateOp>(CallGateOp &,
                                                std::vector<Value> &);
template void classicalCallOperands<CallDefCalGateOp>(CallDefCalGateOp &,
                                                      std::vector<Value> &);
template void classicalCallOperands<CallDefcalMeasureOp>(CallDefcalMeasureOp &,
                                                         std::vector<Value> &);
template void classicalCallOperands<CallSubroutineOp>(CallSubroutineOp &,
                                                      std::vector<Value> &);

// returns the number of qubit arguments for a callOp
template <class CallOpTy>
uint numQubitCallOperands(CallOpTy &callOp) {
  std::vector<Value> tmpVec;
  qubitCallOperands(callOp, tmpVec);
  return tmpVec.size();
} // numQubitCallOperands

// explicit template instantiation
template uint numQubitCallOperands<CallGateOp>(CallGateOp &);
template uint numQubitCallOperands<CallDefCalGateOp>(CallDefCalGateOp &);
template uint numQubitCallOperands<CallDefcalMeasureOp>(CallDefcalMeasureOp &);
template uint numQubitCallOperands<CallSubroutineOp>(CallSubroutineOp &);

// returns the number of classical arguments for a callOp
template <class CallOpTy>
uint numClassicalCallOperands(CallOpTy &callOp) {
  std::vector<Value> tmpVec;
  classicalCallOperands(callOp, tmpVec);
  return tmpVec.size();
} // numQubitCallOperands

// explicit template instantiation
template uint numClassicalCallOperands<CallGateOp>(CallGateOp &);
template uint numClassicalCallOperands<CallDefCalGateOp>(CallDefCalGateOp &);
template uint
numClassicalCallOperands<CallDefcalMeasureOp>(CallDefcalMeasureOp &);
template uint numClassicalCallOperands<CallSubroutineOp>(CallSubroutineOp &);

// appends the indices of the qubit arguments for a callOp to a bit vector
template <class CallOpTy>
void qubitArgIndices(CallOpTy &callOp, llvm::BitVector &bv) {
  uint ii = 0;

  for (auto arg : callOp.getOperands()) {
    if (arg.getType().template isa<QubitType>()) {
      if (bv.size() <= ii)
        bv.resize(ii + 1);
      bv.set(ii);
    }
    ii++;
  }
} // qubitArgIndices

// explicit template instantiation
template void qubitArgIndices<CallGateOp>(CallGateOp &, llvm::BitVector &);
template void qubitArgIndices<CallDefCalGateOp>(CallDefCalGateOp &,
                                                llvm::BitVector &);
template void qubitArgIndices<CallDefcalMeasureOp>(CallDefcalMeasureOp &,
                                                   llvm::BitVector &);
template void qubitArgIndices<CallSubroutineOp>(CallSubroutineOp &,
                                                llvm::BitVector &);

// TODO: Determine a better way of tracing qubit identities
// other than decorating blocks and functions.
std::optional<uint> lookupQubitId(const Value &val) {
  auto declOp = val.getDefiningOp<DeclareQubitOp>();
  if (!declOp) { // Must be an argument to a function
    // see if we can find an attribute with the info
    if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      unsigned const argIdx = blockArg.getArgNumber();
      auto *parentOp = blockArg.getOwner()->getParentOp();
      if (FunctionOpInterface functionOpInterface =
              dyn_cast<FunctionOpInterface>(parentOp)) {
        auto argAttr = functionOpInterface.getArgAttrOfType<IntegerAttr>(
            argIdx, quir::getPhysicalIdAttrName());
        if (argAttr) {
          int id = argAttr.getInt();
          assert(id >= 0 && "physicalId of qubit argument is < 0");
          return static_cast<uint>(id);
        }
      } // if parentOp is funcOp
    }   // if val is blockArg
    return std::nullopt;
  } // if !declOp
  auto idOpt = declOp.getId();

  int id;
  if (idOpt.has_value())
    id = idOpt.value();
  else
    id = -1;

  assert(id >= 0 && "Declared ID of qubit is < 0");
  return static_cast<uint>(id);
} // lookupQubitId

std::optional<Operation *> nextQuantumOpOrNull(Operation *op) {
  Operation *curOp = op;
  while (Operation *nextOp = curOp->getNextNode()) {
    if (isQuantumOp(nextOp))
      return nextOp;
    if (nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      // control flow found, no next quantum op
      return std::nullopt;
    }
    curOp = nextOp;
  }
  return std::nullopt;
} // nextQuantumOpOrNull

std::optional<Operation *> prevQuantumOpOrNull(Operation *op) {
  Operation *curOp = op;
  while (Operation *prevOp = curOp->getPrevNode()) {
    if (isQuantumOp(prevOp))
      return prevOp;
    if (prevOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      // control flow found, no prev quantum op
      return std::nullopt;
    }
    curOp = prevOp;
  }
  return std::nullopt;
} // prevQuantumOpOrNull

std::optional<Operation *> nextQuantumOrControlFlowOrNull(Operation *op) {
  Operation *curOp = op;
  while (Operation *nextOp = curOp->getNextNode()) {
    if (isQuantumOp(nextOp))
      return nextOp;
    if (nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      // control flow found, return
      return nextOp;
    }
    curOp = nextOp;
  }
  return std::nullopt;
} // nextQuantumOrControlFlowOrNull

std::optional<Operation *> prevQuantumOrControlFlowOrNull(Operation *op) {
  Operation *curOp = op;
  while (Operation *prevOp = curOp->getPrevNode()) {
    if (isQuantumOp(prevOp))
      return prevOp;
    if (prevOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      // control flow found, no prev quantum op
      return prevOp;
    }
    curOp = prevOp;
  }
  return std::nullopt;
} // prevQuantumOrControlFlowOrNull

bool isQuantumOp(Operation *op) {
  if (op->hasTrait<mlir::quir::UnitaryOp>() ||
      op->hasTrait<mlir::quir::CPTPOp>() || isa<CallCircuitOp>(op))
    return true;
  return false;
}

llvm::Expected<mlir::quir::DurationAttr>
getDuration(mlir::quir::DelayOp &delayOp) {
  std::string const durationStr;
  auto durationDeclare = delayOp.getTime().getDefiningOp<quir::ConstantOp>();
  if (durationDeclare)
    return durationDeclare.getValue().dyn_cast<quir::DurationAttr>();
  auto argNum = delayOp.getTime().cast<BlockArgument>().getArgNumber();
  auto circuitOp = mlir::dyn_cast<mlir::quir::CircuitOp>(
      delayOp.getTime().getParentBlock()->getParentOp());
  assert(circuitOp && "can only handle circuit arguments");
  auto argAttr = circuitOp.getArgAttrOfType<mlir::quir::DurationAttr>(
      argNum, mlir::quir::getDurationAttrName());
  return argAttr;
}

llvm::Expected<mlir::quir::DurationAttr>
getDuration(mlir::quir::ConstantOp &duration) {
  auto durAttr = duration.getValue().dyn_cast<mlir::quir::DurationAttr>();
  if (!durAttr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Expected a ConstantOp with a DurationAttr");
  return durAttr;
}

std::tuple<Value, MeasureOp> qubitFromMeasResult(MeasureOp measureOp,
                                                 Value result) {
  auto opRes = result.cast<OpResult>();
  uint const resNum = opRes.getResultNumber();
  return std::make_tuple(measureOp.getQubits()[resNum], measureOp);
}

std::tuple<Value, MeasureOp> qubitFromMeasResult(CallCircuitOp callCircuitOp,
                                                 Value result) {
  Operation *findOp =
      SymbolTable::lookupNearestSymbolFrom<mlir::quir::CircuitOp>(
          callCircuitOp, callCircuitOp.getCalleeAttr());

  auto circuitOp = dyn_cast<CircuitOp>(findOp);
  return qubitFromMeasResult(circuitOp, result);
}

std::tuple<Value, MeasureOp> qubitFromMeasResult(CircuitOp circuitOp,
                                                 Value result) {
  auto opRes = result.cast<OpResult>();
  uint const resNum = opRes.getResultNumber();
  auto returnOp = dyn_cast<quir::ReturnOp>(circuitOp.back().getTerminator());
  auto circuitResult = returnOp->getOperand(resNum).cast<OpResult>();
  auto measureOp = dyn_cast<MeasureOp>(circuitResult.getDefiningOp());

  return std::make_tuple(measureOp.getQubits()[circuitResult.getResultNumber()],
                         measureOp);
}

} // end namespace mlir::quir

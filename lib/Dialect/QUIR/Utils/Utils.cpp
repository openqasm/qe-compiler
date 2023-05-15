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
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <regex>

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
    Type t1 = ft1.getInput(ii);
    Type t2 = ft2.getInput(ii);
    if (!quirTypeMatch<AngleType, QubitType>(t1, t2) && t1 != t2)
      return false;
  }

  for (uint ii = 0; ii < ft1.getNumResults(); ++ii) {
    Type t1 = ft1.getResult(ii);
    Type t2 = ft2.getResult(ii);
    if (!quirTypeMatch<AngleType, QubitType>(t1, t2) && t1 != t2)
      return false;
  }

  return true;
} // quirFunctionTypeMatch

// walks the module operation and searches for a func def labeled "main"
auto getMainFunction(Operation *moduleOperation) -> Operation * {
  OpBuilder b(moduleOperation);
  Operation *mainFunc = nullptr;
  moduleOperation->walk([&](FuncOp funcOp) {
    if (SymbolRefAttr::get(funcOp).getLeafReference() == "main") {
      mainFunc = funcOp.getOperation();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
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
auto getNodeId(Operation *moduleOperation) -> int {
  auto typeAttr = moduleOperation->getAttrOfType<IntegerAttr>("quir.nodeId");
  assert(typeAttr && "module Op lacks expected attribute quir.nodeId");
  return typeAttr.getInt();
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
    for (Attribute valAttr : theseIdsAttr) {
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
    for (Attribute valAttr : theseIdsAttr) {
      auto intAttr = valAttr.dyn_cast<IntegerAttr>();
      theseIds.emplace(intAttr.getInt());
    }
  }
} // addQubitIdsFromAttr

// returns a vector of all of the classical arguments for a callOp
template <class CallOpTy>
void classicalCallOperands(CallOpTy &callOp, std::vector<Value> &vec) {
  for (auto arg : callOp.operands())
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

  for (auto arg : callOp.operands()) {
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
llvm::Optional<uint> lookupQubitId(const Value &val) {
  auto declOp = val.getDefiningOp<DeclareQubitOp>();
  if (!declOp) { // Must be an argument to a function
    // see if we can find an attribute with the info
    if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      unsigned argIdx = blockArg.getArgNumber();
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
    return llvm::None;
  } // if !declOp
  int id = declOp.id().getValue();
  assert(id >= 0 && "Declared ID of qubit is < 0");
  return static_cast<uint>(id);
} // lookupQubitId

llvm::Optional<Operation *> nextQuantumOpOrNull(Operation *op) {
  Operation *curOp = op;
  while (Operation *nextOp = curOp->getNextNode()) {
    if (isQuantumOp(nextOp))
      return nextOp;
    if (nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      // control flow found, no next quantum op
      return llvm::None;
    }
    curOp = nextOp;
  }
  return llvm::None;
} // nextQuantumOpOrNull

template <class OpType>
llvm::Optional<OpType> nextQuantumOpOrNullOfType(Operation *op) {
  auto nextOperation = nextQuantumOpOrNull(op);
  if (nextOperation && isa<OpType>(*nextOperation))
    return dyn_cast<OpType>(*nextOperation);
  return llvm::None;
}

// explicit template instantiation
template llvm::Optional<MeasureOp> nextQuantumOpOrNullOfType(Operation *op);

llvm::Optional<Operation *> prevQuantumOpOrNull(Operation *op) {
  Operation *curOp = op;
  while (Operation *prevOp = curOp->getPrevNode()) {
    if (isQuantumOp(prevOp))
      return prevOp;
    if (prevOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      // control flow found, no prev quantum op
      return llvm::None;
    }
    curOp = prevOp;
  }
  return llvm::None;
} // prevQuantumOpOrNull

template <class OpType>
llvm::Optional<OpType> prevQuantumOpOrNullOfType(Operation *op) {
  auto prevOperation = prevQuantumOpOrNull(op);
  if (prevOperation && isa<OpType>(*prevOperation))
    return dyn_cast<OpType>(*prevOperation);
  return llvm::None;
}

// explicit template instantiation
template llvm::Optional<MeasureOp> prevQuantumOpOrNullOfType(Operation *op);

llvm::Optional<Operation *> nextQuantumOrControlFlowOrNull(Operation *op) {
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
  return llvm::None;
} // nextQuantumOrControlFlowOrNull

llvm::Optional<Operation *> prevQuantumOrControlFlowOrNull(Operation *op) {
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
  return llvm::None;
} // prevQuantumOrControlFlowOrNull

bool isQuantumOp(Operation *op) {
  if (op->hasTrait<mlir::quir::UnitaryOp>() ||
      op->hasTrait<mlir::quir::CPTPOp>() || isa<CallCircuitOp>(op))
    return true;
  return false;
}

llvm::Expected<Duration> Duration::parseDuration(mlir::quir::DelayOp &delayOp) {
  std::string durationStr;
  auto durationDeclare = delayOp.time().getDefiningOp<quir::ConstantOp>();
  if (durationDeclare) {
    auto durAttr = durationDeclare.value().dyn_cast<quir::DurationAttr>();
    durationStr = durAttr.getValue().str();
  } else {
    auto argNum = delayOp.time().cast<BlockArgument>().getArgNumber();
    auto circuitOp = mlir::dyn_cast<mlir::quir::CircuitOp>(
        delayOp.time().getParentBlock()->getParentOp());
    assert(circuitOp && "can only handler circuit arguments");
    auto argAttr = circuitOp.getArgAttrOfType<mlir::quir::DurationAttr>(
        argNum, mlir::quir::getDurationAttrName());
    durationStr = argAttr.getValue().str();
  }
  return Duration::parseDuration(durationStr);
}

llvm::Expected<Duration>
Duration::parseDuration(mlir::quir::ConstantOp &duration) {
  auto durAttr = duration.value().dyn_cast<DurationAttr>();
  if (!durAttr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Expected a ConstantOp with a DurationAttr");
  return Duration::parseDuration(durAttr.getValue().str());
}

llvm::Expected<Duration>
Duration::parseDuration(const std::string &durationStr) {
  std::regex re("^([0-9]*[.]?[0-9]+)([a-zA-Z]*)");
  std::smatch m;
  std::regex_match(durationStr, m, re);
  if (m.size() != 3)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine("Unable to parse duration from ") + durationStr);

  double parsedDuration = std::stod(m[1]);
  // Convert all units to lower case.
  auto unitStr = m[2].str();
  auto lowerUnitStr = llvm::StringRef(unitStr).lower();
  DurationUnit parsedUnit;
  if (lowerUnitStr == "") {
    // Empty case is SI
    parsedUnit = DurationUnit::s;
  } else if (lowerUnitStr == "dt") {
    parsedUnit = DurationUnit::dt;
  } else if (lowerUnitStr == "ns") {
    parsedUnit = DurationUnit::ns;
  } else if (lowerUnitStr == "us") {
    parsedUnit = DurationUnit::us;
  } else if (lowerUnitStr == "ms") {
    parsedUnit = DurationUnit::ms;
  } else if (lowerUnitStr == "s") {
    parsedUnit = DurationUnit::s;
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   llvm::Twine("Unknown duration unit ") +
                                       unitStr);
  }

  return (Duration){.duration = parsedDuration, .unit = parsedUnit};
}

Duration Duration::convertToCycles(double dt) const {
  double convertedDuration;

  assert(unit == DurationUnit::dt || unit == DurationUnit::ns ||
         unit == DurationUnit::us || unit == DurationUnit::ms ||
         unit == DurationUnit::s);

  switch (unit) {
  case DurationUnit::dt:
    convertedDuration = duration;
    break;
  case DurationUnit::ns:
    convertedDuration = duration / (1e9 * dt);
    break;
  case DurationUnit::us:
    convertedDuration = duration / (1e6 * dt);
    break;
  case DurationUnit::ms:
    convertedDuration = duration / (1e3 * dt);
    break;
  case DurationUnit::s:
    convertedDuration = duration / dt;
    break;
  }
  return {convertedDuration, DurationUnit::dt};
}

} // end namespace mlir::quir

//===- SubroutineCloning.cpp - Resolve subroutine calls ---------*- C++ -*-===//
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
//  This file implements the pass for cloning subroutines, resolving all qubit
//  arguments and mangling subroutine names with the id of the arguments from
//  each call. Thus each cloned subroutine def matches with a call that has a
//  particular set of qubit arguments. The call is updated to match the newly
//  cloned def.
//
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/SubroutineCloning.h"

#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <sys/types.h>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

auto SubroutineCloningPass::lookupQubitId(const Value val) -> int {
  auto declOp = val.getDefiningOp<DeclareQubitOp>();
  if (declOp) {
    auto id = declOp.getId();
    if (!id.has_value()) {
      declOp->emitOpError() << "Qubit declaration does not have id";
      signalPassFailure();
      return -1;
    }
    return id.value();
  }

  // Must be an argument to a function
  // see if we can find an attribute with the info
  if (auto blockArg = val.dyn_cast<BlockArgument>()) {
    unsigned const argIdx = blockArg.getArgNumber();
    auto funcOp =
        dyn_cast<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (funcOp) {
      auto argAttr = funcOp.getArgAttrOfType<IntegerAttr>(
          argIdx, mlir::quir::getPhysicalIdAttrName());
      if (argAttr)
        return argAttr.getInt();
      funcOp->emitOpError()
          << "does not specify physicalId for qubit argument " << argIdx;
      signalPassFailure();
      return -1;
    } // if (funcOp)
    blockArg.getOwner()->getParentOp()->emitOpError()
        << "Parent op of non-declared qubit value is not a function!?\n";
    signalPassFailure();
    return -1;
  } // if (val is BlockArgument)
  llvm::errs()
      << "Found a qubit that was not declared and is not a block argument\n";
  signalPassFailure();
  return -1;
} // lookupQubitId

template <class CallLikeOp>
auto SubroutineCloningPass::getMangledName(Operation *op) -> std::string {
  auto callOp = dyn_cast<CallLikeOp>(op);
  std::string mangledName = callOp.getCallee().str();

  std::vector<Value> qOperands;
  qubitCallOperands(callOp, qOperands);

  for (Value const qArg : qOperands) {
    int const qId = SubroutineCloningPass::lookupQubitId(qArg);
    if (qId < 0) {
      callOp->emitOpError() << "Unable to resolve qubit ID for call\n";
      callOp->print(llvm::errs());
    }

    mangledName += "_q" + std::to_string(qId);
  }

  return mangledName;
} // getMangledName

template <class CallLikeOp, class FuncLikeOp>
void SubroutineCloningPass::processCallOp(Operation *op,
                                          SymbolOpMap &symbolOps) {
  auto callOp = dyn_cast<CallLikeOp>(op);
  OpBuilder build(moduleOperation->getRegion(0));

  // look for func def match
  auto search = symbolOps.find(callOp.getCallee());

  if (search == symbolOps.end()) {
    callOp->emitOpError() << "No matching function def found for "
                          << callOp.getCallee() << "\n";
    return signalPassFailure();
  }

  Operation *findOp = search->second;
  if (findOp) {
    std::vector<Value> qOperands;
    qubitCallOperands(callOp, qOperands);

    // first get the mangled name
    std::string const mangledName = getMangledName<CallLikeOp>(callOp);
    callOp->setAttr("callee",
                    FlatSymbolRefAttr::get(&getContext(), mangledName));

    // does the mangled function already exist?
    if (symbolOps.find(mangledName) != symbolOps.end())
      return;

    // clone the func def with the new name
    FuncLikeOp newFunc = cast<FuncLikeOp>(build.clone(*findOp));
    newFunc->moveBefore(findOp);
    clonedFuncs.emplace(findOp);
    newFunc->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(&getContext(), mangledName));

    // add qubit ID attributes to all the arguments
    for (uint ii = 0; ii < callOp.getOperands().size(); ++ii) {
      if (callOp.getOperands()[ii].getType().template isa<QubitType>()) {
        int const qId =
            lookupQubitId(callOp.getOperands()[ii]); // copy qubitId from call
        newFunc.setArgAttrs(
            ii, ArrayRef({NamedAttribute(
                    StringAttr::get(&getContext(),
                                    mlir::quir::getPhysicalIdAttrName()),
                    build.getI32IntegerAttr(qId))}));
      }
    }

    // add calls within the new func def to the callWorkList
    newFunc->walk([&](CallLikeOp op) { callWorkList.push_back(op); });

    symbolOps[mangledName] = newFunc.getOperation();

  } else { // matching function not found
    callOp->emitOpError() << "No matching function def found for "
                          << callOp.getCallee() << "\n";
    return signalPassFailure();
  }

} // processCallOp

// Entry point for the pass.
void SubroutineCloningPass::runOnOperation() {
  moduleOperation = getOperation();
  Operation *mainFunc = getMainFunction(moduleOperation);
  callWorkList.clear();

  if (!mainFunc) {
    llvm::errs() << "No main function found, cannot clone subroutines!\n";
    return signalPassFailure();
  }

  mainFunc->walk([&](CallSubroutineOp op) { callWorkList.push_back(op); });

  SymbolOpMap symbolOps;

  if (!callWorkList.empty()) {
    moduleOperation->walk([&](mlir::func::FuncOp functionOp) {
      symbolOps[functionOp.getSymName()] = functionOp.getOperation();
    });
  }

  while (!callWorkList.empty()) {
    Operation *op = callWorkList.front();
    callWorkList.pop_front();
    processCallOp<CallSubroutineOp, mlir::func::FuncOp>(op, symbolOps);
  }

  mainFunc->walk([&](CallCircuitOp op) { callWorkList.push_back(op); });

  if (!callWorkList.empty()) {
    symbolOps.clear();
    moduleOperation->walk([&](CircuitOp circuitOp) {
      symbolOps[circuitOp.getSymName()] = circuitOp.getOperation();
    });
  }

  while (!callWorkList.empty()) {
    Operation *op = callWorkList.front();
    callWorkList.pop_front();
    processCallOp<CallCircuitOp, CircuitOp>(op, symbolOps);
  }

  // All subroutine defs that have been cloned are no longer needed
  for (Operation *op : clonedFuncs)
    op->erase();

} // runOnOperation

llvm::StringRef SubroutineCloningPass::getArgument() const {
  return "subroutine-cloning";
}
llvm::StringRef SubroutineCloningPass::getDescription() const {
  return "Resolve subroutine calls and clone function defs "
         "for each combination of qubit arguments. Calls are updated to match, "
         "the qubit arguments of the cloned def are decorated with attributes "
         "listing the id that the qubit should have";
}

llvm::StringRef SubroutineCloningPass::getName() const {
  return "Subroutine Cloning Pass";
}

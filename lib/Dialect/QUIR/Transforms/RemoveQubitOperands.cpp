//===- RemoveQubitOperands.cpp - Remove qubit args ---------------*- C++-*-===//
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
//  This file implements the pass for removing qubit arguments from subroutines
//  and subroutine calls, replacing the arguments with qubit declarations inside
//  the subroutine body.
//
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/RemoveQubitOperands.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "llvm/ADT/BitVector.h"

using namespace mlir;
using namespace mlir::quir;

auto RemoveQubitOperandsPass::lookupQubitId(const Value val) -> int {
  auto declOp = val.getDefiningOp<DeclareQubitOp>();
  if (declOp)
    return declOp.id().getValue();

  // Must be an argument to a function
  // see if we can find an attribute with the info
  if (auto blockArg = val.dyn_cast<BlockArgument>()) {
    unsigned argIdx = blockArg.getArgNumber();
    auto funcOp = dyn_cast<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (funcOp) {
      auto argAttr = funcOp.getArgAttrOfType<IntegerAttr>(
          argIdx, quir::getPhysicalIdAttrName());
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

void RemoveQubitOperandsPass::addQubitDeclarations(mlir::func::FuncOp funcOp) {
  // build inside the func def body
  OpBuilder build(funcOp.getBody());

  for (auto arg : funcOp.getArguments()) {
    if (arg.getType().isa<QubitType>()) {
      int qId = lookupQubitId(arg);
      if (qId < 0) {
        funcOp->emitOpError()
            << "Subroutine function argument does not specify physicalId, run "
               "subroutine-cloning first\n";
        return signalPassFailure();
      }
      auto newDeclOp = build.create<DeclareQubitOp>(
          funcOp->getLoc(), build.getType<QubitType>(1),
          build.getI32IntegerAttr(qId));
      arg.replaceAllUsesWith(newDeclOp.res());
    }
  }
} // addQubitDeclarations

void RemoveQubitOperandsPass::processCallOp(Operation *op) {
  auto callOp = dyn_cast<CallSubroutineOp>(op);
  llvm::BitVector qIndicesBV(callOp->getNumOperands());

  qubitArgIndices(callOp, qIndicesBV);

  // look for func def match
  Operation *findOp =
      SymbolTable::lookupSymbolIn(moduleOperation, callOp.callee());
  auto funcOp = dyn_cast<mlir::func::FuncOp>(findOp);

  if (!qIndicesBV.empty()) // some qubit args
    callOp->eraseOperands(qIndicesBV);

  if (alreadyProcessed.count(findOp))
    return;

  alreadyProcessed.emplace(findOp);

  if (qIndicesBV.empty()) // no qubit args
    return;

  addQubitDeclarations(funcOp);

  if (qIndicesBV.size() < funcOp.getNumArguments())
    qIndicesBV.resize(funcOp.getNumArguments());
  funcOp.eraseArguments(qIndicesBV);

  findOp->walk([&](CallSubroutineOp op) { callWorkList.push_back(op); });
} // processCallOp

// Entry point for the pass.
void RemoveQubitOperandsPass::runOnOperation() {
  moduleOperation = getOperation();
  Operation *mainFunc = getMainFunction(moduleOperation);
  callWorkList.clear();

  if (!mainFunc) {
    llvm::errs() << "No main function found, cannot remove qubit arguments!\n";
    return signalPassFailure();
  }

  mainFunc->walk([&](CallSubroutineOp op) { callWorkList.push_back(op); });

  while (!callWorkList.empty()) {
    Operation *op = callWorkList.front();
    callWorkList.pop_front();
    processCallOp(op);
  }

  // All subroutine defs that have been cloned are no longer needed
  for (Operation *op : clonedFuncs)
    op->erase();

} // runOnOperation

llvm::StringRef RemoveQubitOperandsPass::getArgument() const {
  return "remove-qubit-args";
}
llvm::StringRef RemoveQubitOperandsPass::getDescription() const {
  return "Remove qubit arguments from subroutine defs and calls, replacing "
         "them with qubit declarations inside the subroutine body";
}

//===- SubroutineCloning.cpp - Resolve subroutine calls ---------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
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
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

using namespace mlir;
using namespace mlir::quir;

auto SubroutineCloningPass::lookupQubitId(const Value val) -> int {
  auto declOp = val.getDefiningOp<DeclareQubitOp>();
  if (declOp)
    return declOp.id().getValue();

  // Must be an argument to a function
  // see if we can find an attribute with the info
  if (auto blockArg = val.dyn_cast<BlockArgument>()) {
    unsigned argIdx = blockArg.getArgNumber();
    auto funcOp = dyn_cast<FuncOp>(blockArg.getOwner()->getParentOp());
    if (funcOp) {
      auto argAttr =
          funcOp.getArgAttrOfType<IntegerAttr>(argIdx, "quir.physicalId");
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

auto SubroutineCloningPass::getMangledName(Operation *op) -> std::string {
  auto callOp = dyn_cast<CallSubroutineOp>(op);
  std::string mangledName = callOp.callee().str();

  std::vector<Value> qArgs;
  qubitCallArgs(callOp, qArgs);

  for (Value qArg : qArgs) {
    int qId = SubroutineCloningPass::lookupQubitId(qArg);
    if (qId < 0) {
      callOp->emitOpError() << "Unable to resolve qubit ID for call\n";
      callOp->print(llvm::errs());
    }

    mangledName += "_q" + std::to_string(qId);
  }

  return mangledName;
} // getMangledName

void SubroutineCloningPass::processCallOp(Operation *op) {
  auto callOp = dyn_cast<CallSubroutineOp>(op);
  OpBuilder build(moduleOperation->getRegion(0));

  // look for func def match
  Operation *findOp =
      SymbolTable::lookupSymbolIn(moduleOperation, callOp.callee());
  if (findOp) {
    std::vector<Value> qArgs;
    qubitCallArgs(callOp, qArgs);

    // first get the mangled name
    std::string mangledName = getMangledName(callOp);
    callOp->setAttr("callee",
                    FlatSymbolRefAttr::get(&getContext(), mangledName));

    // does the mangled function already exist?
    Operation *mangledOp =
        SymbolTable::lookupSymbolIn(moduleOperation, mangledName);
    if (mangledOp) // nothing to do
      return;

    // clone the func def with the new name
    FuncOp newFunc = cast<FuncOp>(build.clone(*findOp));
    newFunc->moveBefore(findOp);
    clonedFuncs.emplace(findOp);
    newFunc->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(&getContext(), mangledName));

    // add qubit ID attributes to all the arguments
    for (uint ii = 0; ii < callOp.args().size(); ++ii) {
      if (callOp.args()[ii].getType().isa<QubitType>()) {
        int qId = lookupQubitId(callOp.args()[ii]); // copy qubitId from call
        newFunc.setArgAttrs(
            ii, ArrayRef({NamedAttribute(
                    StringAttr::get(&getContext(), "quir.physicalId"),
                    build.getI32IntegerAttr(qId))}));
      }
    }

    // add calls within the new func def to the callWorkList
    newFunc->walk([&](CallSubroutineOp op) { callWorkList.push_back(op); });

  } else { // matching function not found
    callOp->emitOpError() << "No matching function def found for "
                          << callOp.callee() << "\n";
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

  while (!callWorkList.empty()) {
    Operation *op = callWorkList.front();
    callWorkList.pop_front();
    processCallOp(op);
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

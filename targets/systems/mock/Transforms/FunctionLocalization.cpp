//===- QUIRFunctionLocalization.cpp - Localizing function defs --*- C++ -*-===//
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
//  This file implements the pass for localizing function defs
//
//===----------------------------------------------------------------------===//

#include "FunctionLocalization.h"

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"

#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "MockFunctionLocalization"

using namespace mlir;
using namespace mlir::quir;
using namespace qssc::targets::systems::mock;

namespace {
// This map is from quir.orig_func_name to a list of functions that have that
// original name
std::unordered_map<std::string, std::list<Operation *>> funcMap;
} // anonymous namespace

void SymbolTableBuildPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  funcMap.clear();
  LLVM_DEBUG(llvm::dbgs() << "\nRunning symbol table build on "
                          << moduleOp.getName() << "\n");

  OpBuilder b(moduleOp);

  // very specifically do not `walk` the region, we only want top-level FuncOps
  for (Region &region : moduleOp.getOperation()->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (auto funcOp = dyn_cast<FuncOp>(op)) {
          std::string fName =
              SymbolRefAttr::get(funcOp).getLeafReference().str();
          auto origNameAttr =
              op.getAttrOfType<StringAttr>("quir.orig_func_name");
          std::string mapName = fName;
          if (origNameAttr)
            mapName = origNameAttr.getValue().str();
          if (funcMap.count(mapName) == 0)
            funcMap.emplace(mapName, std::list(1, &op));
          else
            funcMap[mapName].push_back(&op);
        } // if FuncOp
      }   // for op
    }     // for block
  }       // for region

  LLVM_DEBUG(llvm::dbgs() << "FuncMap:\n"; for (const auto &myPair
                                                : funcMap) {
    llvm::dbgs() << myPair.first << " " << myPair.second.size() << " :\t";
    for (auto *op : myPair.second) {
      auto funcOp = dyn_cast<FuncOp>(op);
      llvm::dbgs() << funcOp.getName() << " ";
    }
    llvm::dbgs() << "\n";
  });

} // SymbolTableBuildPass::runOnOperation

auto MockFunctionLocalizationPass::lookupQubitId(const Value val) -> int {
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

template <class CallOpTy>
auto MockFunctionLocalizationPass::getCallArgIndex(CallOpTy &callOp) -> int {
  int callArgIndex = -1;
  for (uint ii = 0; ii < callOp.operands().size(); ++ii) {
    if (callOp.operands()[ii].getType().template isa<QubitType>()) {
      int qId = lookupQubitId(callOp.operands()[ii]);
      if (qId == theseIds[0]) {
        callArgIndex = ii;
        break;
      }
    }
  }
  return callArgIndex;
} // getCallArgIndex

template <class CallOpTy>
auto MockFunctionLocalizationPass::getMatchedOp(CallOpTy &callOp,
                                                int callArgIndex,
                                                int thisIdIndex)
    -> Operation * {
  std::string calleeName = callOp.callee().str();
  FunctionType cType = callOp.getCalleeType();

  Operation *matchedOp = nullptr;
  if (funcMap.count(calleeName) == 0)
    return matchedOp;

  // This logic will match on the first matching version of the func in the
  // list, however, we should probably do something more intelligent here
  // with some sort of ranking system
  for (auto *op : funcMap[calleeName]) {
    auto funcOp = dyn_cast<FuncOp>(op);
    FunctionType fType = funcOp.getType();
    if (quirFunctionTypeMatch(fType, cType)) { // types match
      auto argIdAttr =
          funcOp.getArgAttrOfType<IntegerAttr>(callArgIndex, "quir.physicalId");
      if (!argIdAttr || thisIdIndex < 0 ||
          (argIdAttr && argIdAttr.getInt() == theseIds[thisIdIndex])) {
        matchedOp = op;
        break;
      }
    } // if types match
  }   // for op in funcMap
  return matchedOp;
} // getMatchedOp

template <class CallOpTy>
auto MockFunctionLocalizationPass::getMangledName(CallOpTy &callOp)
    -> std::string {
  std::string newName = callOp.callee().str();
  for (uint ii = 0; ii < callOp.operands().size(); ++ii) {
    if (callOp.operands()[ii].getType().template isa<QubitType>()) {
      int qId = lookupQubitId(callOp.operands()[ii]);
      newName += "_q" + std::to_string(qId);
    }
  }
  return newName;
} // getMangledName

template <class CallOpTy>
void MockFunctionLocalizationPass::cloneMatchedOp(CallOpTy &callOp,
                                                  const std::string &newName,
                                                  Operation *&clonedOp,
                                                  Operation *matchedOp) {
  // first check if the op has already been copied into this module!
  auto matchedFuncOp = dyn_cast<FuncOp>(matchedOp);
  std::string matchedFuncName =
      SymbolRefAttr::get(matchedFuncOp).getLeafReference().str();
  if (auto *lookedUpOp = SymbolTable::lookupSymbolIn(
          moduleOperation, llvm::StringRef(newName))) {
    clonedOp = lookedUpOp;
  } else { // copy the func def to this module
    clonedOp = builder->clone(*matchedOp);
    auto newFuncOp = dyn_cast<FuncOp>(clonedOp);
    newFuncOp->setAttr(SymbolTable::getSymbolAttrName(),
                       StringAttr::get(newFuncOp.getContext(), newName));
    for (uint ii = 0; ii < callOp.operands().size(); ++ii) {
      if (callOp.operands()[ii].getType().template isa<QubitType>()) {
        int qId =
            lookupQubitId(callOp.operands()[ii]); // copy qubitId from call
        newFuncOp.setArgAttrs(
            ii, ArrayRef({NamedAttribute(
                    StringAttr::get(newFuncOp.getContext(), "quir.physicalId"),
                    builder->getI32IntegerAttr(qId))}));
      }
    }
    toWalk.push_back(clonedOp);
  } // else copy the func def to this module
} // cloneMatchedOp

// Entry point for the pass.
// This should be run on Modules inside of another module
void MockFunctionLocalizationPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  moduleOperation = getOperation();
  Operation *mainFunc = getMainFunction(moduleOp);
  if (!mainFunc) {
    moduleOp->emitOpError()
        << "Error: No main function found, cannot localize functions\n";
    return;
  }

  // insert just before main func def
  builder = std::make_shared<OpBuilder>(mainFunc);

  llvm::StringRef nodeType =
      moduleOp->getAttrOfType<StringAttr>("quir.nodeType").getValue();
  if (nodeType == "controller")
    return;
  LLVM_DEBUG(llvm::dbgs() << "Running function localization on " << nodeType
                          << "\n");

  theseIds.clear();
  auto thisIdAttr = moduleOp->getAttrOfType<IntegerAttr>("quir.physicalId");
  auto theseIdsAttr = moduleOp->getAttrOfType<ArrayAttr>("quir.physicalIds");
  if (thisIdAttr)
    theseIds.push_back(thisIdAttr.getInt());
  if (theseIdsAttr) {
    for (Attribute valAttr : theseIdsAttr) {
      auto intAttr = valAttr.dyn_cast<IntegerAttr>();
      theseIds.push_back(intAttr.getInt());
    }
  }
  if (theseIds.empty()) {
    moduleOp->emitOpError()
        << "Something is wrong in function localization for this node "
        << moduleOp.getName()->str()
        << " doesn't have quir.physicalId or quir.physicalIds attribute!\n";
    signalPassFailure();
    return;
  }

  toWalk.clear();
  toWalk.push_back(mainFunc);

  auto walkSubroutineCalls = [&](CallSubroutineOp callOp) {
    std::string calleeName = callOp.callee().str();
    FunctionType cType = callOp.getCalleeType();

    Operation *matchedOp =
        SymbolTable::lookupSymbolIn(moduleOp, llvm::StringRef(calleeName));
    if (!matchedOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Unable to find func def for " << calleeName << "\n");
      return;
    }
    auto matchedFuncOp = dyn_cast<FuncOp>(matchedOp);
    FunctionType fType = matchedFuncOp.getType();
    if (!quirFunctionTypeMatch(fType, cType)) {
      LLVM_DEBUG(llvm::dbgs() << "The signatures for " << calleeName
                              << " and it's func def don't match!\n");
      return;
    }

    // specialize the function name for this application of the call
    Operation *clonedOp = nullptr;
    std::string newName = getMangledName<CallSubroutineOp>(callOp);
    cloneMatchedOp<CallSubroutineOp>(callOp, newName, clonedOp, matchedOp);

    // now update the callee of the callop
    auto clonedFuncOp = dyn_cast<FuncOp>(clonedOp);
    callOp.calleeAttr(SymbolRefAttr::get(clonedFuncOp));
  }; // walkSubroutineCalls

  auto walkGateCalls = [&](CallGateOp callOp) {
    std::string calleeName = callOp.callee().str();

    int callArgIndex = getCallArgIndex<CallGateOp>(callOp);
    if (callArgIndex ==
        -1) { // This call does not interface with this qubit module
      callOp.getOperation()->erase();
      return;
    }

    Operation *matchedOp = getMatchedOp<CallGateOp>(callOp, callArgIndex, 0);
    if (!matchedOp) { // not found in funcMap!
      // look for callee name in normal symbol table as last resort
      LLVM_DEBUG(llvm::dbgs() << "No match found for " << calleeName << "\n");
      return;
    } // if !match

    // specialize the function name for this application of the call
    Operation *clonedOp = nullptr;
    std::string newName = getMangledName<CallGateOp>(callOp);
    cloneMatchedOp<CallGateOp>(callOp, newName, clonedOp, matchedOp);

    // now update the callee of the callop
    auto clonedFuncOp = dyn_cast<FuncOp>(clonedOp);
    callOp.calleeAttr(SymbolRefAttr::get(clonedFuncOp));
  }; // walkGateCalls

  auto walkDefcalGateCalls = [&](CallDefCalGateOp callOp) {
    std::string calleeName = callOp.callee().str();

    int callArgIndex = getCallArgIndex<CallDefCalGateOp>(callOp);
    if (callArgIndex == -1) {
      // This call does not interface with this qubit module
      callOp.getOperation()->erase();
      return;
    }

    Operation *matchedOp =
        getMatchedOp<CallDefCalGateOp>(callOp, callArgIndex, 0);
    if (!matchedOp) { // not found in funcMap!
      // look for callee name in normal symbol table as last resort
      LLVM_DEBUG(llvm::dbgs() << "No match found for " << calleeName << "\n");
      return;
    } // if !match

    // specialize the function name for this application of the call
    Operation *clonedOp = nullptr;
    std::string newName = getMangledName<CallDefCalGateOp>(callOp);
    cloneMatchedOp<CallDefCalGateOp>(callOp, newName, clonedOp, matchedOp);

    // now update the callee of the callop
    auto clonedFuncOp = dyn_cast<FuncOp>(clonedOp);
    callOp.calleeAttr(SymbolRefAttr::get(clonedFuncOp));
  }; // walkDefcalGateCalls

  auto walkDefcalMeasureCalls = [&](CallDefcalMeasureOp callOp) {
    std::string calleeName = callOp.callee().str();

    int qId = lookupQubitId(callOp.operands()[0]);
    int thisIdIndex = -1;
    for (uint ii = 0; ii < theseIds.size(); ++ii) {
      if (qId == theseIds[ii]) {
        thisIdIndex = ii;
        break;
      }
    }
    if (thisIdIndex == -1) {
      callOp.getOperation()->erase();
      return;
    }

    Operation *matchedOp =
        getMatchedOp<CallDefcalMeasureOp>(callOp, 0, thisIdIndex);
    if (!matchedOp) { // not found in funcMap!
      // look for callee name in normal symbol table as last resort
      LLVM_DEBUG(llvm::dbgs() << "No match found for " << calleeName << "\n");
      return;
    } // if !match

    // found a match
    // specialize the function name for this application of the call
    std::string newName = calleeName + "_q" + std::to_string(qId);

    // first check if the op has already been copied into this module!
    auto matchedFuncOp = dyn_cast<FuncOp>(matchedOp);
    Operation *clonedOp = nullptr;
    std::string matchedFuncName =
        SymbolRefAttr::get(matchedFuncOp).getLeafReference().str();
    if (auto *lookedUpOp =
            SymbolTable::lookupSymbolIn(moduleOp, llvm::StringRef(newName))) {
      clonedOp = lookedUpOp;
    } else { // copy the func def to this module
      clonedOp = builder->clone(*matchedOp);
      auto newFuncOp = dyn_cast<FuncOp>(clonedOp);
      newFuncOp->setAttr(SymbolTable::getSymbolAttrName(),
                         StringAttr::get(newFuncOp.getContext(), newName));
      newFuncOp.setArgAttrs(
          0, ArrayRef({NamedAttribute(
                 StringAttr::get(newFuncOp.getContext(), "quir.physicalId"),
                 builder->getI32IntegerAttr(qId))}));

      toWalk.push_back(clonedOp);
    } // else copy the func def to this module

    // now update the callee of the callop
    auto clonedFuncOp = dyn_cast<FuncOp>(clonedOp);
    callOp.calleeAttr(SymbolRefAttr::get(clonedFuncOp));
  }; // walkDefcalMeasureCalls

  // first walk all subroutine calls
  while (!toWalk.empty()) {
    Operation *walkOp = toWalk.front();
    toWalk.pop_front();
    walkOp->walk(walkSubroutineCalls);
  }

  // now erase unneeded subroutine defs
  bool someErasure = true;
  while (someErasure) {
    someErasure = false;
    std::unordered_set<std::string> subroutineCalls;
    // Find all subroutine calls
    moduleOp->walk([&](CallSubroutineOp callOp) {
      subroutineCalls.emplace(callOp.callee().str());
    });

    moduleOp->walk([&](FuncOp funcOp) {
      llvm::StringRef funcName =
          SymbolRefAttr::get(funcOp.getOperation()).getLeafReference();
      if (subroutineCalls.count(funcName.str()) == 0 && funcName != "main") {
        funcOp.getOperation()->erase();
        someErasure = true; // do it again
      }
    });
  }

  // reset the insertion point if it was corrupted by the erasures
  builder->setInsertionPoint(moduleOp.getBody(), moduleOp.getBody()->begin());

  // refill toWalk
  toWalk.push_back(moduleOp.getOperation());

  // now process gate and measure calls
  while (!toWalk.empty()) {
    Operation *walkOp = toWalk.front();
    toWalk.pop_front();
    walkOp->walk(walkGateCalls);
    walkOp->walk(walkDefcalGateCalls);
    walkOp->walk(walkDefcalMeasureCalls);
  }

} // MockFunctionLocalizationPass::runOnOperation

llvm::StringRef MockFunctionLocalizationPass::getArgument() const {
  return "mock-function-localization";
}

llvm::StringRef MockFunctionLocalizationPass::getDescription() const {
  return "Copy and localize functions for Mock code blocks.";
}

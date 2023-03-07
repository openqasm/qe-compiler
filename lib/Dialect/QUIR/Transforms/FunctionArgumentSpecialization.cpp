//===- FunctionArgumentSpecialization.cpp - Resolve funcs --------*- C++-*-===//
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
//
//  This file implements the pass for specializing function argument widths
//  to match calls. Specialized function defs are cloned with updated argument
//  widths
//
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/FunctionArgumentSpecialization.h"

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "QUIRFunctionArgumentSpecialization"

using namespace mlir;
using namespace mlir::quir;

template <class CallOpTy>
void FunctionArgumentSpecializationPass::processCallOp(
    Operation *op, std::deque<Operation *> &callWorkList) {
  CallOpTy callOp = dyn_cast<CallOpTy>(op);
  if (!callOp) {
    llvm::errs() << "Something really wrong in processCallOp<CalLOpTy>()!\n";
    return;
  }
  // the following line is crazy but it works
  Operation *moduleOp =
      callOp->template getParentOfType<ModuleOp>().getOperation();
  // look for func def match
  Operation *findOp = SymbolTable::lookupSymbolIn(moduleOp, callOp.callee());
  if (findOp) {
    FuncOp funcOp = dyn_cast<FuncOp>(findOp);
    if (!funcOp)
      return;
    // check arguments for width match
    if (!funcOp.getCallableRegion()) {
      // no callable region found (just a prototype)
    } else {
      FunctionType callType = callOp.getCalleeType();
      FunctionType funcType = funcOp.getType();
      if (callType == funcType) {
        // add calls inside this func def to the work list
        findOp->walk([&](Operation *op) {
          if (dyn_cast<CallGateOp>(op) || dyn_cast<CallDefCalGateOp>(op) ||
              dyn_cast<CallDefcalMeasureOp>(op) ||
              dyn_cast<CallSubroutineOp>(op))
            callWorkList.push_back(op);
        });
      } else if (quirFunctionTypeMatch(callType, funcType)) {
        copyFuncAndSpecialize<CallOpTy>(funcOp, callOp, callWorkList);
      } else {
        llvm::errs() << "Fundamental type mismatch between call to "
                     << callOp.callee() << " and func def " << funcOp.getName()
                     << "\n";
      }
    }      // else callable region found
  } else { // callee not matched by FuncOp
    LLVM_DEBUG(llvm::dbgs()
               << "Found call to " << callOp.callee()
               << " with no matching function definition or prototype\n");
  }
} // processCallOp<T1>

template <class T1, class T2, class... Rest>
void FunctionArgumentSpecializationPass::processCallOp(
    Operation *op, std::deque<Operation *> &callWorkList) {
  if (dyn_cast<T1>(op))
    processCallOp<T1>(op, callWorkList);
  else
    processCallOp<T2, Rest...>(op, callWorkList);
} // processCallOp<T1, T2, Rest>

template <class CallOpTy>
void FunctionArgumentSpecializationPass::copyFuncAndSpecialize(
    FuncOp inFunc, CallOpTy callOp, std::deque<Operation *> &callWorkList) {
  OpBuilder b(inFunc);

  std::string newName = SymbolRefAttr::get(inFunc).getLeafReference().str();
  for (auto callOperand : callOp.getOperands()) {
    llvm::raw_string_ostream ss(newName);
    ss << "_" << callOperand.getType();
    newName = ss.str();
  }
  // Check if the specialized function aleady exists
  if (SymbolTable::lookupSymbolIn(inFunc->getParentOp(),
                                  llvm::StringRef(newName))) {
    // function found, nothing to do
    callOp->setAttr("callee", FlatSymbolRefAttr::get(callOp.getContext(),
                                                     llvm::StringRef(newName)));
    return;
  }

  FuncOp newFunc = cast<FuncOp>(b.clone(*inFunc));
  newFunc->moveBefore(inFunc);
  newFunc->setAttr(SymbolTable::getSymbolAttrName(),
                   StringAttr::get(newFunc.getContext(), newName));
  newFunc.setType(callOp.getCalleeType());
  Block &entryBlock = newFunc.front();
  auto callArgTypeIter = callOp.operand_type_begin();
  for (auto blockArg : entryBlock.getArguments()) {
    blockArg.setType(*callArgTypeIter);
    ++callArgTypeIter;
  }

  callOp->setAttr("callee", FlatSymbolRefAttr::get(callOp.getContext(),
                                                   llvm::StringRef(newName)));

  // search for all callOps within the cloned function and add them to the
  // work list
  newFunc.getOperation()->walk([&](Operation *op) {
    if (dyn_cast<CallGateOp>(op) || dyn_cast<CallDefCalGateOp>(op) ||
        dyn_cast<CallDefcalMeasureOp>(op) || dyn_cast<CallSubroutineOp>(op)) {
      callWorkList.push_back(op);
    }
  });
} // copyFuncAndSpecialize

// Entry point for the pass.
void FunctionArgumentSpecializationPass::runOnOperation() {
  // This pass is only called on module Ops
  Operation *moduleOp = getOperation();
  Operation *mainFunc = getMainFunction(moduleOp);
  std::deque<Operation *> callWorkList;

  if (!mainFunc) {
    llvm::errs()
        << "No main function found, cannot specialize function arguments!\n";
    return;
  }

  mainFunc->walk([&](Operation *op) {
    if (dyn_cast<CallGateOp>(op) || dyn_cast<CallDefCalGateOp>(op) ||
        dyn_cast<CallDefcalMeasureOp>(op) || dyn_cast<CallSubroutineOp>(op))
      callWorkList.push_back(op);
  });

  while (!callWorkList.empty()) {
    Operation *op = callWorkList.front();
    callWorkList.pop_front();
    processCallOp<CallGateOp, CallDefCalGateOp, CallDefcalMeasureOp,
                  CallSubroutineOp>(op, callWorkList);
  } // while !callWorkList.empty()
} // runOnOperation

llvm::StringRef FunctionArgumentSpecializationPass::getArgument() const {
  return "quir-arg-specialization";
}

llvm::StringRef FunctionArgumentSpecializationPass::getDescription() const {
  return "Specialize functions defs for quir argument width. Function defs are "
         "cloned, the call is updated to match the cloned def, and the "
         "original function def is left unchanged";
}

//===- MergeCircuits.cpp - Merge call_circuit ops ---------------*- C++ -*-===//
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
///
///  This file implements the pass for merging back to back call_circuits into a
///  single circuit op
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/MergeCircuits.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <llvm/ADT/None.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

namespace {

using MoveListVec = std::vector<Operation *>;
bool moveUsers(Operation *curOp, MoveListVec &moveList) {
  llvm::errs() << "moveUsers start\n";
  moveList.push_back(curOp);
  bool okToMoveUsers = true;
  for (auto user : curOp->getUsers()) {
    llvm::errs() << "User "; user->dump();
    okToMoveUsers &= moveUsers(user, moveList);
  }
  llvm::errs() << "moveUsers end\n";
  return okToMoveUsers;
}

// This pattern matches on two CallCircuitOps separated by non-quantum ops
struct CircuitAndCircuitPattern : public OpRewritePattern<CallCircuitOp> {
  explicit CircuitAndCircuitPattern(MLIRContext *ctx)
      : OpRewritePattern<CallCircuitOp>(ctx) {}

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    // get next quantum op and check if its a CallCircuitOp
    llvm::Optional<Operation *> secondOp = nextQuantumOpOrNull(callCircuitOp);
    if (!secondOp)
      return failure();

    auto nextCallCircuitOp = dyn_cast<CallCircuitOp>(*secondOp);
    if (!nextCallCircuitOp)
      return failure();

#define RUN_MERGE

#ifdef RUN_MERGE
    Operation *insertOp = *secondOp;
#endif

    std::vector<Operation *> moveList;

    // Move first CallCircuitOp after nodes until a user of the
    // CallCircuitOp or the second CallCircuitOp is reached
    Operation *curOp = callCircuitOp->getNextNode();
    while (curOp != *secondOp) {
      llvm::errs() << "curOp "; curOp->dump();
      moveList.clear();
      bool okToMoveUsers = false;
      if (std::find(callCircuitOp->user_begin(), callCircuitOp->user_end(),
                    curOp) != callCircuitOp->user_end()) 
#ifndef RUN_MERGE
        break;
      callCircuitOp->moveAfter(curOp);
#else
                    
      {
        if (std::find(curOp->user_begin(), curOp->user_end(), callCircuitOp) !=
            callCircuitOp->user_end())
          break;
        
        okToMoveUsers = moveUsers(curOp, moveList);
        if (okToMoveUsers)
          for(auto op : moveList) {
            llvm::errs() << "Moving "; op->dump();
            llvm::errs() << "After "; insertOp->dump();

            op->moveAfter(insertOp);
            insertOp = op;
          }
        
      } 
      if (!okToMoveUsers) {
        llvm::errs() << "Not Moving users\n";
        llvm::errs() << "Moving "; callCircuitOp->dump();
        llvm::errs() << "After "; curOp->dump();
        callCircuitOp->moveAfter(curOp);
      }
#endif
      
      curOp = callCircuitOp->getNextNode();
    }

#ifdef RUN_MERGE
    insertOp = nextCallCircuitOp;
#endif

    // Move second CallCircuitOp before nodes until a definition the
    // second CallCircuitOp uses or the first CallCircuitOp is reached
    curOp = nextCallCircuitOp->getPrevNode();
    while (curOp != callCircuitOp) {
      if (std::find(curOp->user_begin(), curOp->user_end(),
                    nextCallCircuitOp) != curOp->user_end())
#ifndef RUN_MERGE
                    
        break;
      nextCallCircuitOp->moveBefore(curOp);
#else
      {
        if (std::find(nextCallCircuitOp->user_begin(),
                      nextCallCircuitOp->user_end(),
                      curOp) != nextCallCircuitOp->user_end())
          break;
        llvm::errs() << "Moving "; curOp->dump();
        llvm::errs() << "Before "; insertOp->dump();
        curOp->moveBefore(insertOp);
        insertOp = curOp;
      } else {
        nextCallCircuitOp->moveBefore(curOp);
      }
#endif
      curOp = nextCallCircuitOp->getPrevNode();
    }

    if (callCircuitOp->getNextNode() != nextCallCircuitOp)
      return failure();

    return MergeCircuitsPass::mergeCallCircuits(rewriter, callCircuitOp,
                                                nextCallCircuitOp);

  } // matchAndRewrite
};  // struct CircuitAndCircuitPattern

template <class FirstOp, class SecondOp>
Optional<SecondOp> getNextOpAndCompareOverlap(FirstOp firstOp) {
  llvm::Optional<Operation *> secondOp = nextQuantumOpOrNull(firstOp);
  if (!secondOp)
    return llvm::None;

  auto secondOpByClass = dyn_cast<SecondOp>(*secondOp);
  if (!secondOpByClass)
    return llvm::None;

  // Check for overlap between currQubits and what's operated on by nextOp
  std::set<uint> firstQubits = QubitOpInterface::getOperatedQubits(firstOp);
  std::set<uint> secondQubits =
      QubitOpInterface::getOperatedQubits(secondOpByClass);

  if (QubitOpInterface::qubitSetsOverlap(firstQubits, secondQubits))
    return llvm::None;
  return secondOpByClass;
}

// This pattern matches on a BarrierOp follows by a CallCircuitOp separated by
// non-quantum ops
struct BarrierAndCircuitPattern : public OpRewritePattern<BarrierOp> {
  explicit BarrierAndCircuitPattern(MLIRContext *ctx)
      : OpRewritePattern<BarrierOp>(ctx) {}

  LogicalResult matchAndRewrite(BarrierOp barrierOp,
                                PatternRewriter &rewriter) const override {

    auto callCircuitOp =
        getNextOpAndCompareOverlap<BarrierOp, CallCircuitOp>(barrierOp);
    if (!callCircuitOp.hasValue())
      return failure();

    barrierOp->moveAfter(callCircuitOp.getValue().getOperation());

    return success();
  } // matchAndRewrite
};  // struct BarrierAndCircuitPattern

// This pattern matches on a CallCircuitOp followed by a BarrierOp separated by
// non-quantum ops
struct CircuitAndBarrierPattern : public OpRewritePattern<CallCircuitOp> {
  explicit CircuitAndBarrierPattern(MLIRContext *ctx)
      : OpRewritePattern<CallCircuitOp>(ctx) {}

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    auto barrierOp =
        getNextOpAndCompareOverlap<CallCircuitOp, BarrierOp>(callCircuitOp);
    if (!barrierOp.hasValue())
      return failure();

    barrierOp.getValue().getOperation()->moveBefore(callCircuitOp);

    return success();
  } // matchAndRewrite
};  // struct CircuitAndBarrierPattern

} // end anonymous namespace

CircuitOp MergeCircuitsPass::getCircuitOp(CallCircuitOp callCircuitOp) {
  auto circuitAttr = callCircuitOp->getAttrOfType<FlatSymbolRefAttr>("callee");
  assert(circuitAttr && "Requires a 'callee' symbol reference attribute");

  auto circuitOp = SymbolTable::lookupNearestSymbolFrom<CircuitOp>(
      callCircuitOp, circuitAttr);
  assert(circuitOp && "matching circuit not found");
  return circuitOp;
}

LogicalResult
MergeCircuitsPass::mergeCallCircuits(PatternRewriter &rewriter,
                                     CallCircuitOp callCircuitOp,
                                     CallCircuitOp nextCallCircuitOp) {
  auto circuitOp = getCircuitOp(callCircuitOp);
  auto nextCircuitOp = getCircuitOp(nextCallCircuitOp);

  OpBuilder builder(circuitOp);
  builder.setInsertionPointAfter(nextCircuitOp);

  llvm::SmallVector<Type> inputTypes;
  llvm::SmallVector<Value> inputValues;
  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;

  // merge input type into single SmallVector
  inputTypes.append(circuitOp->getOperandTypes().begin(),
                    circuitOp->getOperandTypes().end());
  inputTypes.append(nextCircuitOp->getOperandTypes().begin(),
                    nextCircuitOp->getOperandTypes().end());

  // merge circuit names
  std::string newName =
      (circuitOp.sym_name() + "_" + nextCircuitOp.sym_name()).str();

  // create new circuit operation by cloning first circuit
  CircuitOp newCircuitOp = cast<CircuitOp>(builder.clone(*circuitOp));
  newCircuitOp->setAttr(SymbolTable::getSymbolAttrName(),
                        StringAttr::get(circuitOp->getContext(), newName));

  // store original return operations for later use
  quir::ReturnOp returnOp;
  quir::ReturnOp nextReturnOp;

  circuitOp->walk([&](quir::ReturnOp r) { returnOp = r; });
  nextCircuitOp->walk([&](quir::ReturnOp r) { nextReturnOp = r; });

  // map original arguments for new circuit based on original circuit
  // argument numbers
  BlockAndValueMapping mapper;
  auto baseArgNum = newCircuitOp.getNumArguments();
  for (uint cnt = 0; cnt < nextCircuitOp.getNumArguments(); cnt++) {
    auto arg = nextCircuitOp.getArgument(cnt);
    auto dictArg = nextCircuitOp.getArgAttrDict(cnt);
    newCircuitOp.insertArgument(baseArgNum + cnt, arg.getType(), dictArg,
                                arg.getLoc());
    mapper.map(arg, newCircuitOp.getArgument(baseArgNum + cnt));
  }

  // find return op in new circuit and copy second circuit into the
  // new circuit
  quir::ReturnOp newReturnOp;
  newCircuitOp->walk([&](quir::ReturnOp r) { newReturnOp = r; });
  builder.setInsertionPointAfter(newReturnOp);

  for (auto &block : nextCircuitOp.getBody().getBlocks())
    for (auto &op : block.getOperations())
      builder.clone(op, mapper);

  // remove any existing return operations from new circuit
  // collect their output types and values into vectors
  newCircuitOp->walk([&](quir::ReturnOp r) {
    outputValues.append(r.getOperands().begin(), r->getOperands().end());
    outputTypes.append(r->getOperandTypes().begin(),
                       r->getOperandTypes().end());
    r->erase();
  });

  // create a return op in the new circuit with the merged output values
  OpBuilder newBuilder(&newCircuitOp.back().back());
  newBuilder.setInsertionPointAfter(&newCircuitOp.back().back());
  newBuilder.create<quir::ReturnOp>(nextReturnOp->getLoc(), outputValues);

  // change the input / output types for the quir.circuit
  auto opType = newCircuitOp.getType();
  newCircuitOp.setType(builder.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  // merge the physical ID attributes
  auto theseIdsAttr = newCircuitOp->getAttrOfType<ArrayAttr>(
      mlir::quir::getPhysicalIdsAttrName());

  auto newIdsAttr = nextCircuitOp->getAttrOfType<ArrayAttr>(
      mlir::quir::getPhysicalIdsAttrName());

  // collect all of the first circuit's ids
  std::vector<int> allIds;
  for (Attribute valAttr : theseIdsAttr) {
    auto intAttr = valAttr.dyn_cast<IntegerAttr>();
    allIds.push_back(intAttr.getInt());
  }

  // add IDs from the second circuit if not from the first
  for (Attribute valAttr : newIdsAttr) {
    auto intAttr = valAttr.dyn_cast<IntegerAttr>();
    auto result = std::find(begin(allIds), end(allIds), intAttr.getInt());
    if (result == end(allIds))
      allIds.push_back(intAttr.getInt());
  }

  newCircuitOp->setAttr(mlir::quir::getPhysicalIdsAttrName(),
                        builder.getI32ArrayAttr(ArrayRef<int>(allIds)));

  // merge the call_circuits
  // collect their input values
  llvm::SmallVector<Value> callInputValues;
  callInputValues.append(callCircuitOp->getOperands().begin(),
                         callCircuitOp.getOperands().end());
  callInputValues.append(nextCallCircuitOp->getOperands().begin(),
                         nextCallCircuitOp.getOperands().end());

  // replace or create new based on original outputType sizes
  if (nextCallCircuitOp->getNumResults() == outputTypes.size()) {
    rewriter.replaceOpWithNewOp<CallCircuitOp>(nextCallCircuitOp, newName,
                                               TypeRange(outputTypes),
                                               ValueRange(callInputValues));
    rewriter.eraseOp(callCircuitOp);
  } else if (callCircuitOp->getNumResults() == outputTypes.size()) {
    rewriter.replaceOpWithNewOp<CallCircuitOp>(callCircuitOp, newName,
                                               TypeRange(outputTypes),
                                               ValueRange(callInputValues));
    rewriter.eraseOp(nextCallCircuitOp);
  } else {
    // can not directly replace a call circuit since the number of results does
    // not match

    auto numCallResults = callCircuitOp->getNumResults();

    auto newCallOp = rewriter.create<mlir::quir::CallCircuitOp>(
        callCircuitOp->getLoc(), newName, TypeRange(outputTypes),
        ValueRange(callInputValues));
    for (const auto &res : llvm::enumerate(callCircuitOp->getResults()))
      res.value().replaceAllUsesWith(newCallOp.getResult(res.index()));
    for (const auto &res : llvm::enumerate(nextCallCircuitOp->getResults())) {
      res.value().replaceAllUsesWith(
          newCallOp.getResult(res.index() + numCallResults));
    }
    callCircuitOp->erase();
    nextCallCircuitOp->erase();
  }

  return success();
}

void MergeCircuitsPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<CircuitAndCircuitPattern>(&getContext());
  patterns.add<BarrierAndCircuitPattern>(&getContext());
  patterns.add<CircuitAndBarrierPattern>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();
} // runOnOperation

llvm::StringRef MergeCircuitsPass::getArgument() const {
  return "merge-circuits";
}
llvm::StringRef MergeCircuitsPass::getDescription() const {
  return "Merge back-to-back call_circuits ";
}

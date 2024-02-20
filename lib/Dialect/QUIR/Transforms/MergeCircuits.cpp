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

#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <set>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

llvm::cl::list<std::string>
    disabledPatterns("merge-circuits-disabled-patterns",
                     llvm::cl::desc("Patterns to disable from merge-circuits"));

namespace {

using MoveListVec = std::vector<Operation *>;
bool moveUsers(Operation *curOp, MoveListVec &moveList) {
  moveList.push_back(curOp);
  for (auto *user : curOp->getUsers())
    if (!moveUsers(user, moveList))
      return false;
  return true;
}

// This pattern matches on two CallCircuitOps separated by non-quantum ops
struct CircuitAndCircuitPattern : public OpRewritePattern<CallCircuitOp> {
  explicit CircuitAndCircuitPattern(MLIRContext *ctx,
                                    llvm::StringMap<Operation *> &symbolMap)
      : OpRewritePattern<CallCircuitOp>(ctx) {
    _symbolMap = &symbolMap;
  }

  llvm::StringMap<Operation *> *_symbolMap;

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    // find next CallCircuitOp or fail
    Operation *searchOp = callCircuitOp.getOperation();
    std::optional<Operation *> secondOp;
    CallCircuitOp nextCallCircuitOp;
    while (true) {
      secondOp = nextQuantumOpOrNull(searchOp);
      if (!secondOp)
        return failure();

      nextCallCircuitOp = dyn_cast<CallCircuitOp>(*secondOp);
      if (nextCallCircuitOp)
        break;

      // check for overlap in qubits between the circuit and the
      // next quantum circuit which is not a CallCircuit
      // fail if there is overlap
      std::set<uint> firstQubits =
          QubitOpInterface::getOperatedQubits(callCircuitOp);
      std::set<uint> secondQubits =
          QubitOpInterface::getOperatedQubits(*secondOp);

      if (QubitOpInterface::qubitSetsOverlap(firstQubits, secondQubits))
        return failure();

      searchOp = *secondOp;
    }

    Operation *insertOp = *secondOp;

    MoveListVec moveList;

    // Move first CallCircuitOp after nodes until a user of the
    // CallCircuitOp or the second CallCircuitOp is reached
    Operation *curOp = callCircuitOp->getNextNode();
    while (curOp != *secondOp) {
      moveList.clear();
      bool okToMoveUsers = false;

      // do not attempt to merge circuits with a parallel control flow op in
      // between
      if (isa<qcs::ParallelControlFlowOp>(curOp))
        return failure();

      if (std::find(callCircuitOp->user_begin(), callCircuitOp->user_end(),
                    curOp) != callCircuitOp->user_end())

      {
        if (std::find(curOp->user_begin(), curOp->user_end(), callCircuitOp) !=
            callCircuitOp->user_end())
          break;

        okToMoveUsers = moveUsers(curOp, moveList);
        if (okToMoveUsers)
          for (auto *op : moveList) {
            op->moveAfter(insertOp);
            insertOp = op;
          }
      }
      if (!okToMoveUsers)
        callCircuitOp->moveAfter(curOp);

      curOp = callCircuitOp->getNextNode();
    }

    insertOp = nextCallCircuitOp;

    // Move second CallCircuitOp before nodes until a definition the
    // second CallCircuitOp uses or the first CallCircuitOp is reached
    curOp = nextCallCircuitOp->getPrevNode();
    while (curOp != callCircuitOp) {
      if (std::find(curOp->user_begin(), curOp->user_end(),
                    nextCallCircuitOp) != curOp->user_end()) {
        if (std::find(nextCallCircuitOp->user_begin(),
                      nextCallCircuitOp->user_end(),
                      curOp) != nextCallCircuitOp->user_end())
          break;
        curOp->moveBefore(insertOp);
        insertOp = curOp;
      } else {
        nextCallCircuitOp->moveBefore(curOp);
      }
      curOp = nextCallCircuitOp->getPrevNode();
    }

    if (callCircuitOp->getNextNode() != nextCallCircuitOp)
      return failure();

    return MergeCircuitsPass::mergeCallCircuits(
        getContext(), rewriter, callCircuitOp, nextCallCircuitOp, _symbolMap);
  } // matchAndRewrite

  void initialize() { setDebugName("CircuitAndCircuitPattern"); }
}; // struct CircuitAndCircuitPattern

template <class FirstOp, class SecondOp>
std::optional<SecondOp> getNextOpAndCompareOverlap(FirstOp firstOp) {
  std::optional<Operation *> secondOp = nextQuantumOpOrNull(firstOp);
  if (!secondOp)
    return std::nullopt;

  auto secondOpByClass = dyn_cast<SecondOp>(*secondOp);
  if (!secondOpByClass)
    return std::nullopt;

  // Check for overlap between currQubits and what's operated on by nextOp
  std::set<uint> firstQubits = QubitOpInterface::getOperatedQubits(firstOp);
  std::set<uint> secondQubits =
      QubitOpInterface::getOperatedQubits(secondOpByClass);

  if (QubitOpInterface::qubitSetsOverlap(firstQubits, secondQubits))
    return std::nullopt;
  return secondOpByClass;
}

// This pattern matches on a BarrierOp follows by a CallCircuitOp separated by
// non-quantum ops
struct BarrierAndCircuitPattern : public OpRewritePattern<BarrierOp> {
  explicit BarrierAndCircuitPattern(MLIRContext *ctx)
      : OpRewritePattern<BarrierOp>(ctx) {}

  LogicalResult matchAndRewrite(BarrierOp barrierOp,
                                PatternRewriter &rewriter) const override {

    // check for circuit op to merge with after moving barrier
    auto prevCallCircuitOp =
        prevQuantumOpOrNullOfType<CallCircuitOp>(barrierOp);
    if (!prevCallCircuitOp)
      return failure();

    auto callCircuitOp =
        getNextOpAndCompareOverlap<BarrierOp, CallCircuitOp>(barrierOp);
    if (!callCircuitOp.has_value())
      return failure();

    barrierOp->moveAfter(callCircuitOp.value().getOperation());

    return success();
  } // matchAndRewrite

  void initialize() { setDebugName("BarrierAndCircuitPattern"); }
}; // struct BarrierAndCircuitPattern

// This pattern matches on a CallCircuitOp followed by a BarrierOp separated by
// non-quantum ops
struct CircuitAndBarrierPattern : public OpRewritePattern<CallCircuitOp> {
  explicit CircuitAndBarrierPattern(MLIRContext *ctx)
      : OpRewritePattern<CallCircuitOp>(ctx) {}

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    auto barrierOp =
        getNextOpAndCompareOverlap<CallCircuitOp, BarrierOp>(callCircuitOp);
    if (!barrierOp.has_value())
      return failure();

    auto *barrierOperation = barrierOp.value().getOperation();
    // check for circuit op to merge with
    auto nextCallCircuitOp =
        nextQuantumOpOrNullOfType<CallCircuitOp>(barrierOperation);
    if (!nextCallCircuitOp)
      return failure();

    barrierOperation->moveBefore(callCircuitOp);

    return success();
  } // matchAndRewrite

  void initialize() { setDebugName("CircuitAndBarrierPattern"); }
}; // struct CircuitAndBarrierPattern

// This pattern matches on a CallCircuitOp, multiple barriers, CallCircuitOp
struct CircuitBarrierCircuitPattern : public OpRewritePattern<CallCircuitOp> {
  explicit CircuitBarrierCircuitPattern(MLIRContext *ctx,
                                        llvm::StringMap<Operation *> &symbolMap)
      : OpRewritePattern<CallCircuitOp>(ctx) {
    _symbolMap = &symbolMap;
  }

  llvm::StringMap<Operation *> *_symbolMap;

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    Operation *curOp = callCircuitOp->getNextNode();
    llvm::SmallVector<Operation *> barrierOps;
    while (isa<BarrierOp>(curOp)) {
      barrierOps.push_back(curOp);
      curOp = curOp->getNextNode();
    };

    if (barrierOps.size() == 0)
      return failure();

    auto nextCallCircuitOp = dyn_cast<CallCircuitOp>(curOp);
    if (!nextCallCircuitOp)
      return failure();

    return MergeCircuitsPass::mergeCallCircuits(
        getContext(), rewriter, callCircuitOp, nextCallCircuitOp, _symbolMap,
        barrierOps);
    return success();
  } // matchAndRewrite

  void initialize() { setDebugName("CircuitBarrierCircuitPattern"); }
}; // struct CircuitAndBarrierPattern

} // end anonymous namespace

CircuitOp
MergeCircuitsPass::getCircuitOp(CallCircuitOp callCircuitOp,
                                llvm::StringMap<Operation *> *symbolMap) {
  // look for func def match
  assert(symbolMap && "a valid symbolMap pointer must be provided");
  auto search = symbolMap->find(callCircuitOp.getCallee());

  assert(search != symbolMap->end() && "matching circuit not found");

  auto circuitOp = dyn_cast<CircuitOp>(search->second);
  assert(circuitOp && "matching circuit not found");
  return circuitOp;
}

uint insertOperandAndArgument(
    Value operand, Operation *defOp, llvm::SmallVector<Value> &callInputValues,
    std::unordered_map<Operation *, uint> &inputValueIndices,
    uint insertedCount) {
  auto argumentIndexIter = inputValueIndices.find(defOp);
  uint argumentIndex = 0;
  if (argumentIndexIter == inputValueIndices.end()) {
    argumentIndex = insertedCount;
    callInputValues.push_back(operand);
    inputValueIndices[defOp] = argumentIndex;
  } else {
    argumentIndex = argumentIndexIter->second;
  }
  return argumentIndex;
}

void mapNextCircuitOperands(
    CallCircuitOp nextCallCircuitOp, CircuitOp nextCircuitOp,
    CircuitOp newCircuitOp, llvm::SmallVector<Value> &callInputValues,
    std::unordered_map<Operation *, uint> &inputValueIndices,
    IRMapping &mapper) {

  // insert nextCircuit Operands into callInputValues if not there
  // add arguments as well
  uint insertedCount = inputValueIndices.size();
  for (auto operandEnum : llvm::enumerate(nextCallCircuitOp.getOperands())) {
    auto arg = nextCircuitOp.getArgument(operandEnum.index());
    auto *defOp = operandEnum.value().getDefiningOp();
    auto argumentIndex =
        insertOperandAndArgument(operandEnum.value(), defOp, callInputValues,
                                 inputValueIndices, insertedCount);
    if (argumentIndex == insertedCount) {
      auto dictArg = nextCircuitOp.getArgAttrDict(operandEnum.index());
      newCircuitOp.insertArgument(argumentIndex, arg.getType(), dictArg,
                                  arg.getLoc());
      insertedCount++;
    }
    mapper.map(arg, newCircuitOp.getArgument(argumentIndex));
  }
}

void mapBarrierOperands(
    Operation *barrierOp, CircuitOp newCircuitOp,
    llvm::SmallVector<Value> &callInputValues,
    std::unordered_map<Operation *, uint> &inputValueIndices, IRMapping &mapper,
    MLIRContext *context) {
  assert(barrierOp && "barrierOp requires valid operation pointer");
  assert(context && "context requires valid MLIR context pointer");
  uint insertedCount = inputValueIndices.size();
  for (auto operand : barrierOp->getOperands()) {
    auto *defOp = operand.getDefiningOp();
    auto argumentIndex = insertOperandAndArgument(
        operand, defOp, callInputValues, inputValueIndices, insertedCount);
    if (argumentIndex == insertedCount) {
      auto physicalId = defOp->getAttrOfType<IntegerAttr>("id");
      newCircuitOp.insertArgument(argumentIndex, operand.getType(), {},
                                  operand.getLoc());
      newCircuitOp.setArgAttrs(
          argumentIndex,
          ArrayRef({NamedAttribute(
              StringAttr::get(context, mlir::quir::getPhysicalIdAttrName()),
              physicalId)}));
      insertedCount++;
    }
    mapper.map(operand, newCircuitOp.getArgument(argumentIndex));
  }
}

void mergePhysicalIdAttrs(CircuitOp newCircuitOp, CircuitOp nextCircuitOp,
                          PatternRewriter &rewriter) {
  // merge the physical ID attributes
  auto theseIdsAttr = newCircuitOp->getAttrOfType<ArrayAttr>(
      mlir::quir::getPhysicalIdsAttrName());

  auto newIdsAttr = nextCircuitOp->getAttrOfType<ArrayAttr>(
      mlir::quir::getPhysicalIdsAttrName());

  // collect all of the first circuit's ids
  std::vector<int> allIds;
  for (Attribute const valAttr : theseIdsAttr) {
    auto intAttr = valAttr.dyn_cast<IntegerAttr>();
    allIds.push_back(intAttr.getInt());
  }

  // add IDs from the second circuit if not from the first
  for (Attribute const valAttr : newIdsAttr) {
    auto intAttr = valAttr.dyn_cast<IntegerAttr>();
    auto result = std::find(begin(allIds), end(allIds), intAttr.getInt());
    if (result == end(allIds))
      allIds.push_back(intAttr.getInt());
  }

  newCircuitOp->setAttr(mlir::quir::getPhysicalIdsAttrName(),
                        rewriter.getI32ArrayAttr(ArrayRef<int>(allIds)));
}

quir::ReturnOp getReturnOp(Operation *op) {
  quir::ReturnOp returnOp;
  op->walk([&](quir::ReturnOp r) { returnOp = r; });
  return returnOp;
}

LogicalResult MergeCircuitsPass::mergeCallCircuits(
    MLIRContext *context, PatternRewriter &rewriter,
    CallCircuitOp callCircuitOp, CallCircuitOp nextCallCircuitOp,
    llvm::StringMap<Operation *> *symbolMap,
    std::optional<llvm::SmallVector<Operation *>> barrierOps) {
  auto circuitOp = getCircuitOp(callCircuitOp, symbolMap);
  auto nextCircuitOp = getCircuitOp(nextCallCircuitOp, symbolMap);

  rewriter.setInsertionPointAfter(nextCircuitOp);

  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;

  // merge the call_circuits
  // collect their input values
  llvm::SmallVector<Value> callInputValues;
  std::unordered_map<Operation *, uint> inputValueIndices;
  for (auto inputValueEnum : llvm::enumerate(callCircuitOp->getOperands())) {
    auto *defOp = inputValueEnum.value().getDefiningOp();
    callInputValues.push_back(inputValueEnum.value());
    inputValueIndices[defOp] = inputValueEnum.index();
  }

  // merge circuit names
  std::string const newName =
      (circuitOp.getSymName() + "_" + nextCircuitOp.getSymName()).str();

  // create new circuit operation by cloning first circuit
  CircuitOp newCircuitOp = cast<CircuitOp>(rewriter.clone(*circuitOp));
  newCircuitOp->setAttr(SymbolTable::getSymbolAttrName(),
                        StringAttr::get(circuitOp->getContext(), newName));

  // store original return operations for later use
  quir::ReturnOp const nextReturnOp = getReturnOp(nextCircuitOp);

  IRMapping mapper;

  // map original arguments for new circuit based on original circuit
  // argument numbers
  mapNextCircuitOperands(nextCallCircuitOp, nextCircuitOp, newCircuitOp,
                         callInputValues, inputValueIndices, mapper);

  // find return op in new circuit
  quir::ReturnOp const newReturnOp = getReturnOp(newCircuitOp);
  rewriter.setInsertionPointAfter(newReturnOp);

  // clone any barrier ops and erase
  if (barrierOps.has_value()) {
    for (auto *barrierOp : barrierOps.value()) {
      // add barrierOps to argument list for circuit and set physicalId
      // of attribute
      mapBarrierOperands(barrierOp, newCircuitOp, callInputValues,
                         inputValueIndices, mapper, context);
      // clone into circuit and remove from original location
      rewriter.clone(*barrierOp, mapper);
      barrierOp->erase();
    }
  }

  // copy second circuit into the new circuit
  for (auto &block : nextCircuitOp.getBody().getBlocks())
    for (auto &op : block.getOperations())
      rewriter.clone(op, mapper);

  // remove any existing return operations from new circuit
  // collect their output types and values into vectors
  newCircuitOp->walk([&](quir::ReturnOp r) {
    outputValues.append(r.getOperands().begin(), r->getOperands().end());
    outputTypes.append(r->getOperandTypes().begin(),
                       r->getOperandTypes().end());
    rewriter.eraseOp(r);
  });

  // create a return op in the new circuit with the merged output values
  rewriter.setInsertionPointToEnd(&newCircuitOp.back());
  rewriter.create<quir::ReturnOp>(nextReturnOp->getLoc(), outputValues);

  // change the input / output types for the quir.circuit
  auto opType = newCircuitOp.getFunctionType();
  newCircuitOp.setType(rewriter.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  mergePhysicalIdAttrs(newCircuitOp, nextCircuitOp, rewriter);

  rewriter.setInsertionPointAfter(nextCallCircuitOp);
  auto newCallOp = rewriter.create<mlir::quir::CallCircuitOp>(
      callCircuitOp->getLoc(), newName, TypeRange(outputTypes),
      ValueRange(callInputValues));

  // dice the output so we can specify which results to replace
  auto iterSep = newCallOp.result_begin() + callCircuitOp.getNumResults();
  rewriter.replaceOp(callCircuitOp,
                     ResultRange(newCallOp.result_begin(), iterSep));
  rewriter.replaceOp(nextCallCircuitOp,
                     ResultRange(iterSep, newCallOp.result_end()));

  // add new name to symbolMap
  // do not remove old in case the are multiple calls
  (*symbolMap)[newName] = newCircuitOp.getOperation();

  return success();
}

void MergeCircuitsPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  llvm::StringMap<Operation *> circuitOpsMap;

  moduleOperation->walk([&](CircuitOp circuitOp) {
    circuitOpsMap[circuitOp.getSymName()] = circuitOp.getOperation();
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<CircuitAndCircuitPattern>(&getContext(), circuitOpsMap);
  patterns.add<BarrierAndCircuitPattern>(&getContext());
  patterns.add<CircuitAndBarrierPattern>(&getContext());
  patterns.add<CircuitBarrierCircuitPattern>(&getContext(), circuitOpsMap);

  // default enable / disable patters
  SmallVector<std::string> const enabledPatterns = {};
  auto frozenPatterns = FrozenRewritePatternSet(
      std::move(patterns), disabledPatterns, enabledPatterns);

  mlir::GreedyRewriteConfig config;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  if (failed(applyPatternsAndFoldGreedily(moduleOperation, frozenPatterns,
                                          config)))
    signalPassFailure();
} // runOnOperation

llvm::StringRef MergeCircuitsPass::getArgument() const {
  return "merge-circuits";
}
llvm::StringRef MergeCircuitsPass::getDescription() const {
  return "Merge back-to-back call_circuits ";
}

llvm::StringRef MergeCircuitsPass::getName() const {
  return "Merge Circuits Pass";
}

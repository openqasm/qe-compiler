//===- MergeCircuitMeasures.cpp - Merge measurements in circuits *- C++ -*-===//
//
// (C) Copyright IBM 2024.
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
///  This file implements the pass for merging measurements located in circuits
///  into a single measure op
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/MergeCircuitMeasures.h"

#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <set>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

namespace {

static std::string duplicateCircuit(PatternRewriter &rewriter,
                                    CircuitOp circuitOp,
                                    llvm::StringMap<Operation *> &symbolMap,
                                    const std::string &newNameTemplate,
                                    const std::string &salt) {
  rewriter.setInsertionPoint(circuitOp);
  auto *oldCircuitOp = rewriter.clone(*circuitOp);
  symbolMap[circuitOp.getSymName()] = oldCircuitOp;

  // merge circuit names with an additional salt for the merge

  std::string newName = newNameTemplate + salt;
  std::string testName = newName;
  int cnt = 0;
  while (symbolMap.contains(testName))
    testName = newName + std::to_string(cnt++);
  newName = testName;

  circuitOp->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(circuitOp->getContext(), newName));
  symbolMap[newName] = circuitOp;

  return newName;
}

static void initTypeAndValVectors(std::vector<Type> &typeVec,
                                  std::vector<Value> &valVec,
                                  MeasureOp measureOp,
                                  MeasureOp nextMeasureOp) {
  typeVec.reserve(measureOp.getNumResults() + nextMeasureOp.getNumResults());
  valVec.reserve(measureOp.getNumResults() + nextMeasureOp.getNumResults());

  typeVec.insert(typeVec.end(), measureOp.result_type_begin(),
                 measureOp.result_type_end());
  typeVec.insert(typeVec.end(), nextMeasureOp.result_type_begin(),
                 nextMeasureOp.result_type_end());
  valVec.insert(valVec.end(), measureOp.getQubits().begin(),
                measureOp.getQubits().end());
}

static void remapArguments(PatternRewriter &rewriter,
                           std::vector<Value> &valVec,
                           CallCircuitOp callCircuitOp,
                           CallCircuitOp nextCallCircuitOp, CircuitOp circuitOp,
                           CircuitOp nextCircuitOp, MeasureOp nextMeasureOp) {

  // remap measurement arguments
  // - build list of arguments to map to
  std::unordered_map<uint32_t, BlockArgument> circuitArguments;
  for (uint argNum = 0; argNum < circuitOp.getNumArguments(); argNum++) {
    auto argAttr = circuitOp.getArgAttrOfType<IntegerAttr>(
        argNum, mlir::quir::getPhysicalIdAttrName());
    circuitArguments[argAttr.getInt()] = circuitOp.getArgument(argNum);
  }

  auto maxArgument = circuitOp.getNumArguments();

  auto theseIdsAttr =
      circuitOp->getAttrOfType<ArrayAttr>(mlir::quir::getPhysicalIdsAttrName());

  // collect all of the first circuit's ids
  std::vector<int> allIds;
  for (Attribute const valAttr : theseIdsAttr) {
    auto intAttr = valAttr.dyn_cast<IntegerAttr>();
    allIds.push_back(intAttr.getInt());
  }

  // remap arguments
  for (auto argument : nextMeasureOp.getQubits()) {
    auto blockArgument = dyn_cast<BlockArgument>(argument);
    assert(blockArgument && "expect qubit to be a block argument");
    auto argNum = blockArgument.getArgNumber();
    auto argAttr = nextCircuitOp.getArgAttrOfType<IntegerAttr>(
        argNum, mlir::quir::getPhysicalIdAttrName());
    auto search = circuitArguments.find(argAttr.getInt());
    if (search == circuitArguments.end()) {
      auto operand = nextCallCircuitOp.getOperand(argNum);
      callCircuitOp->insertOperands(maxArgument, ValueRange{operand});
      auto dictArg = nextCircuitOp.getArgAttrDict(argNum);
      circuitOp.insertArgument(maxArgument, argument.getType(), dictArg,
                               argument.getLoc());
      auto newArg = circuitOp.getArgument(maxArgument);
      circuitArguments[argAttr.getInt()] = newArg;
      valVec.push_back(newArg);
      allIds.push_back(argAttr.getInt());
      maxArgument++;
    } else {
      valVec.push_back(search->second);
    }
  }

  circuitOp->setAttr(mlir::quir::getPhysicalIdsAttrName(),
                     rewriter.getI32ArrayAttr(ArrayRef<int>(allIds)));
}

static void buildOutputs(llvm::SmallVector<Type> &outputTypes,
                         llvm::SmallVector<Value> &outputValues,
                         quir::ReturnOp returnOp, unsigned int resultOffset,
                         MeasureOp mergedOp) {
  outputTypes.append(returnOp->getOperandTypes().begin(),
                     returnOp->getOperandTypes().end());

  outputValues.append(returnOp->getOperands().begin(),
                      returnOp->getOperands().end());

  auto result = mergedOp.getOuts().begin() + resultOffset;
  for (; result != mergedOp.getResults().end(); result++) {
    outputTypes.push_back((*result).getType());
    outputValues.push_back(*result);
  }
}

static CallCircuitOp
createNewCallOp(PatternRewriter &rewriter, llvm::SmallVector<Type> &outputTypes,
                llvm::SmallVector<Value> &outputValues, quir::ReturnOp returnOp,
                CallCircuitOp callCircuitOp, CircuitOp circuitOp,
                llvm::StringRef newName1) {
  rewriter.setInsertionPointAfter(returnOp);
  auto newReturnOp =
      rewriter.create<quir::ReturnOp>(returnOp->getLoc(), outputValues);
  // newReturnOp.dump();
  rewriter.replaceOp(returnOp, newReturnOp->getResults());

  // change the input / output types for the quir.circuit
  auto opType = circuitOp.getFunctionType();
  circuitOp.setType(rewriter.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  // new call circuit ops
  rewriter.setInsertionPointAfter(callCircuitOp);
  auto newCallOp = rewriter.create<mlir::quir::CallCircuitOp>(
      callCircuitOp->getLoc(), newName1, TypeRange(outputTypes),
      ValueRange(callCircuitOp->getOperands()));
  return newCallOp;
}

static void dropNextMeasure(PatternRewriter &rewriter,
                            llvm::SmallVector<Type> &outputTypes,
                            llvm::SmallVector<Value> &outputValues,
                            CircuitOp nextCircuitOp, MeasureOp nextMeasureOp) {
  auto nextReturnOp = dyn_cast<quir::ReturnOp>(&nextCircuitOp.back().back());
  assert(nextReturnOp && "quir.circuit must end end a quir.return");
  outputTypes.clear();
  outputValues.clear();
  std::vector<int> eraseList;
  for (uint idx = 0; idx < nextReturnOp.getNumOperands(); idx++)
    if (nextReturnOp->getOperand(idx).getDefiningOp() ==
        nextMeasureOp.getOperation())
      eraseList.push_back(idx);

  while (!eraseList.empty()) {
    nextReturnOp->eraseOperand(eraseList.back());
    eraseList.pop_back();
  }

  rewriter.eraseOp(nextMeasureOp);

  outputTypes.append(nextReturnOp->getOperandTypes().begin(),
                     nextReturnOp->getOperandTypes().end());

  auto opType = nextCircuitOp.getFunctionType();
  nextCircuitOp.setType(rewriter.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));
}

static void mergeMeasurements(PatternRewriter &rewriter,
                              CallCircuitOp callCircuitOp,
                              CallCircuitOp nextCallCircuitOp,
                              CircuitOp circuitOp, CircuitOp nextCircuitOp,
                              MeasureOp measureOp, MeasureOp nextMeasureOp,
                              llvm::StringMap<Operation *> &symbolMap) {

  // copy circuitOp in case there are multiple calls
  std::string const newNameTemplate =
      (circuitOp.getSymName() + "_" + nextCircuitOp.getSymName()).str();
  auto newName1 =
      duplicateCircuit(rewriter, circuitOp, symbolMap, newNameTemplate, "+m");

  // copy nextCircuitOp in case there are multiple calls
  auto newName2 = duplicateCircuit(rewriter, nextCircuitOp, symbolMap,
                                   newNameTemplate, "-m");

  // merge measurements
  std::vector<Type> typeVec;
  std::vector<Value> valVec;
  initTypeAndValVectors(typeVec, valVec, measureOp, nextMeasureOp);

  remapArguments(rewriter, valVec, callCircuitOp, nextCallCircuitOp, circuitOp,
                 nextCircuitOp, nextMeasureOp);

  // find return and update
  auto returnOp = dyn_cast<quir::ReturnOp>(&circuitOp.back().back());
  assert(returnOp && "quir.circuit must end end a quir.return");

  rewriter.setInsertionPoint(measureOp);
  auto mergedOp = rewriter.create<MeasureOp>(
      measureOp.getLoc(), TypeRange(typeVec), ValueRange(valVec));

  auto originalNumResults = measureOp->getNumResults();
  rewriter.replaceOp(measureOp, ResultRange(mergedOp.getOuts().begin(),
                                            mergedOp.getOuts().end()));

  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;

  buildOutputs(outputTypes, outputValues, returnOp, originalNumResults,
               mergedOp);

  auto newCallOp =
      createNewCallOp(rewriter, outputTypes, outputValues, returnOp,
                      callCircuitOp, circuitOp, newName1);

  dropNextMeasure(rewriter, outputTypes, outputValues, nextCircuitOp,
                  nextMeasureOp);

  // dice the output so we can specify which results to replace
  auto iterSep = newCallOp.result_begin() + callCircuitOp.getNumResults();
  rewriter.replaceOp(callCircuitOp,
                     ResultRange(newCallOp.result_begin(), iterSep));
  rewriter.replaceOp(nextCallCircuitOp,
                     ResultRange(iterSep, newCallOp.result_end()));

  // delete the nextCircuit if it is now empty (starts with a return)
  auto firstReturnOp = dyn_cast<quir::ReturnOp>(&nextCircuitOp.front().front());
  if (firstReturnOp) {
    symbolMap.erase(nextCircuitOp.getSymName());
    rewriter.eraseOp(nextCircuitOp);
  }
}

// This pattern matches on two MeasureOps that are only interspersed by
// classical non-control flow ops and merges them into one measure op
struct CallCircuitAndCallCircuitTopologicalPattern
    : public OpRewritePattern<CallCircuitOp> {
  explicit CallCircuitAndCallCircuitTopologicalPattern(
      MLIRContext *ctx, llvm::StringMap<Operation *> &symbolMap)
      : OpRewritePattern<CallCircuitOp>(ctx) {
    _symbolMap = &symbolMap;
  }

  llvm::StringMap<Operation *> *_symbolMap;

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    // check to see if CallCircuit is inside a function and ignore as
    // functions do not label their qubit physical ids this causes later
    // getOperatedQubits to fail
    auto funcOp = dyn_cast<func::FuncOp>(callCircuitOp->getParentOp());
    if (funcOp && (SymbolRefAttr::get(funcOp).getLeafReference() != "main"))
      return failure();

    // is there a measure in the current circuit
    auto search1 = _symbolMap->find(callCircuitOp.getCallee());
    assert(search1 != _symbolMap->end() && "matching circuit not found");

    auto firstCircuit = dyn_cast<CircuitOp>(search1->second);

    MeasureOp firstMeasureOp;
    std::set<uint32_t> currMeasureQubits;

    auto *firstOp = &firstCircuit.getBody().front().front();

    if (isa<MeasureOp>(firstOp)) {
      firstMeasureOp = dyn_cast<MeasureOp>(firstOp);
      currMeasureQubits = QubitOpInterface::getOperatedQubits(firstMeasureOp);
    } else {
      auto firstMeasureOpt =
          QubitOpInterface::getNextQubitOpOfType<MeasureOp>(firstOp);
      if (!firstMeasureOpt.has_value())
        return failure();
      firstMeasureOp = firstMeasureOpt.value();
      currMeasureQubits = firstMeasureOp.getOperatedQubits();
    }

    auto [nextCallCircuitOp, observedQubits] =
        QubitOpInterface::getNextQubitOpOfTypeWithQubits<CallCircuitOp>(
            callCircuitOp);

    if (!nextCallCircuitOp.has_value())
      return failure();

    auto search2 = _symbolMap->find(nextCallCircuitOp->getCallee());
    assert(search2 != _symbolMap->end() && "matching circuit not found");

    auto secondCircuit = dyn_cast<CircuitOp>(search2->second);

    // Find the next measurement operation accumulating qubits along the
    // topological path if it exists

    firstOp = &secondCircuit.getBody().front().front();

    MeasureOp nextMeasureOp;

    if (isa<MeasureOp>(firstOp)) {
      nextMeasureOp = dyn_cast<MeasureOp>(firstOp);
    } else {
      auto [nextMeasureOpt, additionalObservedQubits] =
          QubitOpInterface::getNextQubitOpOfTypeWithQubits<MeasureOp>(firstOp);

      observedQubits.insert(additionalObservedQubits.begin(),
                            additionalObservedQubits.end());
      if (!nextMeasureOpt.has_value())
        return failure();
      nextMeasureOp = nextMeasureOpt.value();
    }

    // If any qubit along path touches the same qubits we cannot merge the next
    // measurement.
    currMeasureQubits.insert(observedQubits.begin(), observedQubits.end());

    // found a measure and a measure, now make sure they aren't working on the
    // same qubit and that we can resolve them both
    auto nextMeasureQubits = nextMeasureOp.getOperatedQubits();

    // If there is an intersection we cannot merge
    std::set<int> mergeMeasureIntersection;
    std::set_intersection(currMeasureQubits.begin(), currMeasureQubits.end(),
                          nextMeasureQubits.begin(), nextMeasureQubits.end(),
                          std::inserter(mergeMeasureIntersection,
                                        mergeMeasureIntersection.begin()));

    if (!mergeMeasureIntersection.empty())
      return failure();

    // good to merge
    mergeMeasurements(rewriter, callCircuitOp, *nextCallCircuitOp, firstCircuit,
                      secondCircuit, firstMeasureOp, nextMeasureOp,
                      *_symbolMap);

    return success();
  } // matchAndRewrite
};  // struct CircuitAndCircuitTopologicalPattern
} // end anonymous namespace

void MergeCircuitMeasuresTopologicalPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  llvm::StringMap<Operation *> circuitOpsMap;

  moduleOperation->walk([&](CircuitOp circuitOp) {
    circuitOpsMap[circuitOp.getSymName()] = circuitOp.getOperation();
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<CallCircuitAndCallCircuitTopologicalPattern>(&getContext(),
                                                            circuitOpsMap);

  mlir::GreedyRewriteConfig config;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  if (failed(applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns),
                                          config)))
    signalPassFailure();
} // runOnOperation

llvm::StringRef MergeCircuitMeasuresTopologicalPass::getArgument() const {
  return "merge-circuit-measures-topological";
}
llvm::StringRef MergeCircuitMeasuresTopologicalPass::getDescription() const {
  return "Merge qubit-parallel measurement operations inside circuits into a "
         "single measurement operation with topological ordering";
}

llvm::StringRef MergeCircuitMeasuresTopologicalPass::getName() const {
  return "Merge Circuit Measures Topological Pass";
}

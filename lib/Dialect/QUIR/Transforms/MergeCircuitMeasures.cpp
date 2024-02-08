//===- MergeMeasures.cpp - Merge measurement ops ----------------*- C++ -*-===//
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
///  This file implements the pass for merging measurements into a single
///  measure op
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/MergeCircuitMeasures.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"

#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <set>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

namespace {

static void mergeMeasurements(PatternRewriter &rewriter,
                              CallCircuitOp callCircuitOp,
                              CallCircuitOp nextCallCircuitOp,
                              CircuitOp circuitOp, CircuitOp nextCircuitOp,
                              MeasureOp measureOp, MeasureOp nextMeasureOp,
                              llvm::StringMap<Operation *> &symbolMap) {

  // copy circuitOp in case there are multiple calls
  rewriter.setInsertionPoint(circuitOp);
  rewriter.clone(*circuitOp);
  symbolMap[nextCircuitOp.getSymName()] = circuitOp;

  // merge circuit names with an additional m for the merge
  std::string const newName1 =
      (circuitOp.getSymName() + "_" + nextCircuitOp.getSymName()).str() + "+m";

  // rename first circuit to the new name
  circuitOp->setAttr(SymbolTable::getSymbolAttrName(),
                     StringAttr::get(circuitOp->getContext(), newName1));
  symbolMap[newName1] = circuitOp;

  // copy nextCircuitOp in case there are multiple calls
  rewriter.setInsertionPoint(nextCircuitOp);
  rewriter.clone(*nextCircuitOp);
  symbolMap[nextCircuitOp.getSymName()] = nextCircuitOp;

  // merge circuit names with an additional m for the merge
  auto newName = nextCircuitOp.getSymName().str() + "-m";
  nextCircuitOp->setAttr(SymbolTable::getSymbolAttrName(),
                         StringAttr::get(circuitOp->getContext(), newName));
  symbolMap[newName] = nextCircuitOp;

  // merge measurements
  std::vector<Type> typeVec;
  std::vector<Value> valVec;
  typeVec.reserve(measureOp.getNumResults() + nextMeasureOp.getNumResults());
  valVec.reserve(measureOp.getNumResults() + nextMeasureOp.getNumResults());

  typeVec.insert(typeVec.end(), measureOp.result_type_begin(),
                 measureOp.result_type_end());
  typeVec.insert(typeVec.end(), nextMeasureOp.result_type_begin(),
                 nextMeasureOp.result_type_end());
  valVec.insert(valVec.end(), measureOp.getQubits().begin(),
                measureOp.getQubits().end());

  // remap measurement arguments
  // - build list of arguments to map to
  std::unordered_map<uint32_t, BlockArgument> circuitArguments;
  for (uint argNum = 0; argNum < circuitOp.getNumArguments(); argNum++) {
    auto argAttr = circuitOp.getArgAttrOfType<IntegerAttr>(
        argNum, mlir::quir::getPhysicalIdAttrName());
    circuitArguments[argAttr.getInt()] = circuitOp.getArgument(argNum);
    circuitArguments[argAttr.getInt()].dump();
  }

  auto maxArgument = circuitOp.getNumArguments();

  // // - remap arguments
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
      maxArgument++;
    } else {
      valVec.push_back(search->second);
    }
  }

  // find return and update
  auto returnOp = dyn_cast<quir::ReturnOp>(&circuitOp.back().back());
  assert(returnOp && "quir.circuit must end end a quir.return");

  rewriter.setInsertionPoint(measureOp);
  auto mergedOp = rewriter.create<MeasureOp>(
      measureOp.getLoc(), TypeRange(typeVec), ValueRange(valVec));

  returnOp->dump();

  rewriter.replaceOp(measureOp, ResultRange(mergedOp.getOuts().begin(),
                                            mergedOp.getOuts().end()));

  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;

  outputTypes.append(returnOp->getOperandTypes().begin(),
                     returnOp->getOperandTypes().end());

  outputValues.append(returnOp->getOperands().begin(),
                      returnOp->getOperands().end());

  auto resultType =
      mergedOp->getResultTypes().begin() + measureOp.getNumResults() + 1;
  for (; resultType != mergedOp->getResultTypes().end(); ++resultType)
    outputTypes.push_back(*resultType);

  auto result = mergedOp->getResults().begin() + measureOp.getNumResults() + 1;
  for (; result != mergedOp->getResults().end(); ++result)
    outputValues.push_back(*result);

  rewriter.setInsertionPointAfter(returnOp);
  auto newReturnOp =
      rewriter.create<quir::ReturnOp>(returnOp->getLoc(), outputValues);
  newReturnOp.dump();
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

  // drop nextMeasure
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

  opType = nextCircuitOp.getFunctionType();
  nextCircuitOp.setType(rewriter.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  // dice the output so we can specify which results to replace
  auto iterSep = newCallOp.result_begin() + callCircuitOp.getNumResults();
  rewriter.replaceOp(callCircuitOp,
                     ResultRange(newCallOp.result_begin(), iterSep));
  rewriter.replaceOp(nextCallCircuitOp,
                     ResultRange(iterSep, newCallOp.result_end()));

  llvm::errs() << "First Circuit:\n";
  circuitOp.dump();

  llvm::errs() << "Second Circuit:\n";
  nextCircuitOp.dump();

  llvm::errs() << "\n\n\n\n\n";

  // delete the nextCircuit if it is now empty (starts with a return)
  auto firstReturnOp = dyn_cast<quir::ReturnOp>(&nextCircuitOp.front().front());
  if (firstReturnOp)
    rewriter.eraseOp(nextCircuitOp);
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

    // Two types of measurements to merge
    // 1. Two measurements in the same circuit (possibly covered by
    // MergeMeasuresTopologicalPass)
    // 2. Two measurements in different circuits

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

    llvm::errs() << "First Circuit: \n";
    firstCircuit.dump();

    llvm::errs() << "First Measure: \n";
    firstMeasureOp.dump();

    llvm::errs() << "Operated Qubits = \n";
    for (auto qubit : currMeasureQubits)
      llvm::errs() << qubit << "\n";

    llvm::errs() << "Second Circuit: \n";
    secondCircuit.dump();

    llvm::errs() << "Next Measure: \n";
    nextMeasureOp->dump();

    llvm::errs() << "Observed Qubits = \n";
    for (auto qubit : observedQubits)
      llvm::errs() << qubit << "\n";

    llvm::errs() << "Found circuits to try an merge\n";

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

    llvm::errs() << "Test Intersection = \n";
    for (auto qubit : mergeMeasureIntersection)
      llvm::errs() << qubit << "\n";

    if (!mergeMeasureIntersection.empty())
      return failure();

    // good to merge
    llvm::errs() << "Good to Merge\n";
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

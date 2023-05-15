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

#include <vector>

using namespace mlir;
using namespace mlir::quir;

namespace {

// This pattern matches on two CallCircuitOp back to back
struct CircuitAndCircuitPattern : public OpRewritePattern<CallCircuitOp> {
  explicit CircuitAndCircuitPattern(MLIRContext *ctx)
      : OpRewritePattern<CallCircuitOp>(ctx) {}

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    // get next operation and test for Delay
    Operation *nextOp = callCircuitOp->getNextNode();
    if (!nextOp)
      return failure();

    auto nextCallCircuitOp = dyn_cast<CallCircuitOp>(nextOp);
    if (!nextCallCircuitOp)
      return failure();

    return MergeCircuitsPass::mergeCallCircuits(rewriter, callCircuitOp,
                                                nextCallCircuitOp);

  } // matchAndRewrite
};  // struct CircuitAndCircuitPattern
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
  llvm::Twine newName = circuitOp.sym_name() + "_" + nextCircuitOp.sym_name();

  // create new circuit operation by cloning first circuit
  CircuitOp newCircuitOp = cast<CircuitOp>(builder.clone(*circuitOp));
  newCircuitOp->setAttr(
      SymbolTable::getSymbolAttrName(),
      StringAttr::get(circuitOp->getContext(), newName.str()));

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

  // replace or create new based on origina outputType sizes
  if (nextCallCircuitOp->getNumResults() == outputTypes.size()) {
    rewriter.replaceOpWithNewOp<CallCircuitOp>(nextCallCircuitOp, newName.str(),
                                               TypeRange(outputTypes),
                                               ValueRange(callInputValues));
    rewriter.eraseOp(callCircuitOp);
  } else if (callCircuitOp->getNumResults() == outputTypes.size()) {
    rewriter.replaceOpWithNewOp<CallCircuitOp>(callCircuitOp, newName.str(),
                                               TypeRange(outputTypes),
                                               ValueRange(callInputValues));
    rewriter.eraseOp(nextCallCircuitOp);
  } else {
    // can not directly replace a call circuit since the number of results does
    // not match

    auto numCallResults = callCircuitOp->getNumResults();

    auto newCallOp = rewriter.create<mlir::quir::CallCircuitOp>(
        callCircuitOp->getLoc(), newName.str(), TypeRange(outputTypes),
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

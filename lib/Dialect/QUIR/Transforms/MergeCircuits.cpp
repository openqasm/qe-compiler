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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <deque>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <unordered_set>
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

    auto moduleOp = callCircuitOp->getParentOfType<ModuleOp>();

    // moduleOp.dump();

    // good to merge
    MergeCircuitsPass::mergeCallCircuits(rewriter, callCircuitOp,
                                         nextCallCircuitOp);

    // moduleOp.dump();

    return success();
  } // matchAndRewrite
};  // struct CircuitAndCircuitPattern
} // end anonymous namespace

CircuitOp MergeCircuitsPass::getCircuitOp(CallCircuitOp callCircuitOp) {
  auto circuitAttr = callCircuitOp->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!circuitAttr) {
    callCircuitOp.emitOpError("Requires a 'callee' symbol reference attribute");
    // signalPassFailure();
  }

  auto circuitOp = SymbolTable::lookupNearestSymbolFrom<CircuitOp>(
      callCircuitOp, circuitAttr);
  if (!circuitOp) {
    callCircuitOp.emitOpError() << "'" << circuitAttr.getValue()
                                << "' does not reference a valid circuit";
    // signalPassFailure();
  }
  return circuitOp;
}

void MergeCircuitsPass::mergeCallCircuits(PatternRewriter &rewriter,
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

  inputTypes.append(circuitOp->getOperandTypes().begin(),
                    circuitOp->getOperandTypes().end());
  inputTypes.append(nextCircuitOp->getOperandTypes().begin(),
                    nextCircuitOp->getOperandTypes().end());

  llvm::Twine newName = circuitOp.sym_name() + "_" + nextCircuitOp.sym_name();

  CircuitOp newCircuitOp = cast<CircuitOp>(builder.clone(*circuitOp));
  newCircuitOp->setAttr(
      SymbolTable::getSymbolAttrName(),
      StringAttr::get(circuitOp->getContext(), newName.str()));

  quir::ReturnOp returnOp;
  quir::ReturnOp nextReturnOp;

  BlockAndValueMapping mapper;

  circuitOp->walk([&](quir::ReturnOp r) { returnOp = r; });
  nextCircuitOp->walk([&](quir::ReturnOp r) { nextReturnOp = r; });

  auto baseArgNum = newCircuitOp.getNumArguments();

  for (uint cnt = 0; cnt < nextCircuitOp.getNumArguments(); cnt++) {
    auto arg = nextCircuitOp.getArgument(cnt);
    auto dictArg = nextCircuitOp.getArgAttrDict(cnt);
    newCircuitOp.insertArgument(baseArgNum + cnt, arg.getType(), dictArg,
                                arg.getLoc());
    mapper.map(arg, newCircuitOp.getArgument(baseArgNum + cnt));
  }

  quir::ReturnOp newReturnOp;
  newCircuitOp->walk([&](quir::ReturnOp r) { newReturnOp = r; });
  builder.setInsertionPointAfter(newReturnOp);

  for (auto &block : nextCircuitOp.getBody().getBlocks())
    for (auto &op : block.getOperations())
      builder.clone(op, mapper);

  newCircuitOp->walk([&](quir::ReturnOp r) {
    outputValues.append(r.getOperands().begin(), r->getOperands().end());
    outputTypes.append(r->getOperandTypes().begin(),
                       r->getOperandTypes().end());
    r->erase();
  });

  OpBuilder newBuilder(&newCircuitOp.back().back());
  newBuilder.setInsertionPointAfter(&newCircuitOp.back().back());
  newBuilder.create<quir::ReturnOp>(nextReturnOp->getLoc(), outputValues);

  auto opType = newCircuitOp.getType();

  // change the input / output types for the quir.circuit
  newCircuitOp.setType(builder.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  auto theseIdsAttr = newCircuitOp->getAttrOfType<ArrayAttr>(
      mlir::quir::getPhysicalIdsAttrName());

  auto newIdsAttr = nextCircuitOp->getAttrOfType<ArrayAttr>(
      mlir::quir::getPhysicalIdsAttrName());

  std::vector<int> allIds;

  for (Attribute valAttr : theseIdsAttr) {
    auto intAttr = valAttr.dyn_cast<IntegerAttr>();
    allIds.push_back(intAttr.getInt());
  }

  for (Attribute valAttr : newIdsAttr) {
    auto intAttr = valAttr.dyn_cast<IntegerAttr>();
    auto result = std::find(begin(allIds), end(allIds), intAttr.getInt());
    if (result == end(allIds))
      allIds.push_back(intAttr.getInt());
  }

  newCircuitOp->setAttr(mlir::quir::getPhysicalIdsAttrName(),
                        builder.getI32ArrayAttr(ArrayRef<int>(allIds)));

  llvm::SmallVector<Value> callInputValues;

  callInputValues.append(callCircuitOp->getOperands().begin(),
                         callCircuitOp.getOperands().end());
  callInputValues.append(nextCallCircuitOp->getOperands().begin(),
                         nextCallCircuitOp.getOperands().end());

  // auto newCallOp = rewriter.create<mlir::quir::CallCircuitOp>(
  //     callCircuitOp->getLoc(), newName.str(),
  //     TypeRange(outputTypes), ValueRange(callInputValues));

  rewriter.replaceOpWithNewOp<CallCircuitOp>(nextCallCircuitOp, newName.str(),
                                             TypeRange(outputTypes),
                                             ValueRange(callInputValues));
  rewriter.eraseOp(callCircuitOp);
  // nextCallCircuitOp->erase();
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

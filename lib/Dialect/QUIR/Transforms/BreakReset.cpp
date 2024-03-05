//===- BreakReset.cpp - Break apart reset ops -------------------*- C++ -*-===//
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
///  This file implements the pass for breaking apart reset ops into
///  control flow.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/BreakReset.h"

#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIREnums.h"
#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

namespace {
// This pattern merges qubit reset operations that can be parallelized into one
// operation.
struct BreakResetsPattern : public OpRewritePattern<ResetQubitOp> {

  explicit BreakResetsPattern(MLIRContext *ctx, uint numIterations,
                              uint delayCycles, BreakResetPass &thisPass)
      : OpRewritePattern<ResetQubitOp>(ctx), numIterations_(numIterations),
        delayCycles_(delayCycles), thisPass_(thisPass) {}

  LogicalResult matchAndRewrite(ResetQubitOp resetOp,
                                PatternRewriter &rewriter) const override {
    quir::ConstantOp constantDurationOp;

    if (auto circuitOp = resetOp->getParentOfType<CircuitOp>()) {
      llvm::errs() << "Need to handle reset in circuitOp\n";
      // TODO: implement a strategy for breaking resets inside of
      // currently the QUIRGenQASM3Visitor does not put resets
      // inside of circuits to prevent problems with this pass.
      return failure();
    }

    if (numIterations_ > 1 && delayCycles_ > 0) {
      constantDurationOp = rewriter.create<quir::ConstantOp>(
          resetOp.getLoc(),
          DurationAttr::get(rewriter.getContext(),
                            rewriter.getType<DurationType>(TimeUnits::dt),
                            /* cast to int first to address ambiguity in uint
                               cast across platforms */
                            llvm::APFloat(static_cast<double>(
                                static_cast<int64_t>(delayCycles_)))));
    }

    // result of measurement in each iteration is number of qubits * i1
    std::vector<mlir::Type> const typeVec(resetOp.getQubits().size(),
                                          rewriter.getI1Type());

    for (uint iteration = 0; iteration < numIterations_; iteration++) {
      if (delayCycles_ > 0 && iteration > 0)
        for (auto qubit : resetOp.getQubits())
          rewriter.create<DelayOp>(resetOp.getLoc(),
                                   constantDurationOp.getResult(), qubit);

      auto measureOp = rewriter.create<MeasureOp>(
          resetOp.getLoc(), TypeRange(typeVec), resetOp.getQubits());
      measureOp->setAttr(getNoReportRuntimeAttrName(), rewriter.getUnitAttr());

      if (thisPass_.insertQuantumGatesIntoCirc)
        thisPass_.measureList.push_back(measureOp);

      size_t i = 0;
      for (auto qubit : resetOp.getQubits()) {
        auto ifOp = rewriter.create<scf::IfOp>(resetOp.getLoc(),
                                               measureOp.getResult(i), false);
        auto savedInsertionPoint = rewriter.saveInsertionPoint();
        auto *thenBlock = ifOp.getBody(0);

        rewriter.setInsertionPointToStart(thenBlock);
        auto callGateOp = rewriter.create<CallGateOp>(
            resetOp.getLoc(), StringRef("x"), TypeRange{}, ValueRange{qubit});
        if (thisPass_.insertQuantumGatesIntoCirc)
          thisPass_.callGateList.push_back(callGateOp);

        i++;
        rewriter.restoreInsertionPoint(savedInsertionPoint);
      }
    }

    rewriter.eraseOp(resetOp);
    return success();
  }

private:
  uint numIterations_;
  uint delayCycles_;
  BreakResetPass &thisPass_;
}; // BreakResetsPattern
} // anonymous namespace

void BreakResetPass::runOnOperation() {
  // check for command line override of insertQuantumGatesIntoCirc
  if (insertCallGatesAndMeasuresIntoCircuit.hasValue())
    insertQuantumGatesIntoCirc =
        insertCallGatesAndMeasuresIntoCircuit.getValue();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::GreedyRewriteConfig config;

  // use cheaper top-down traversal (in this case, bottom-up would not behave
  // any differently)
  config.useTopDownTraversal = true;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  patterns.add<BreakResetsPattern>(&getContext(), numIterations, delayCycles,
                                   *this);

  if (mlir::failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config)))
    signalPassFailure();

  if (insertQuantumGatesIntoCirc) {
    mlir::ModuleOp moduleOp = getOperation();
    assert(moduleOp && "cannot find module op");
    // populate symbol map for input circuits
    moduleOp->walk([&](Operation *op) {
      if (auto castOp = dyn_cast<CircuitOp>(op))
        inputCircuitsSymbolMap[castOp.getSymName()] = castOp.getOperation();
    });

    // insert measures and call gates into circuits -- when
    // insertCallGatesAndMeasuresIntoCircuit option is true
    while (!measureList.empty()) {
      const MeasureOp measOp = dyn_cast<MeasureOp>(measureList.front());
      insertMeasureInCircuit(moduleOp, measOp);
      measureList.pop_front();
    }
    while (!callGateList.empty()) {
      const CallGateOp callGateOp = dyn_cast<CallGateOp>(callGateList.front());
      insertCallGateInCircuit(moduleOp, callGateOp);
      callGateList.pop_front();
    }
  }
} // BreakResetPass::runOnOperation

void BreakResetPass::insertCallGateInCircuit(
    ModuleOp moduleOp, mlir::quir::CallGateOp callGateOp) {
  std::vector<Value> qubitOperands;
  qubitCallOperands(callGateOp, qubitOperands);
  std::string circuitName = "reset_x";
  assert(qubitOperands.size() == 1 && "x gate call have only 1 qubit operand");
  auto declOp = qubitOperands[0].getDefiningOp<quir::DeclareQubitOp>();
  assert(declOp && "cannot find the declare qubit op");
  auto qubitId = quir::lookupQubitId(declOp);
  assert(qubitId.has_value() && "cannot find the qubit id");
  circuitName += ("_" + std::to_string(qubitId.value()));

  auto search = resetGateCircuitsSymbolMap.find(circuitName);
  CircuitOp circOp;
  if (search == resetGateCircuitsSymbolMap.end()) {
    // build a circuit
    circOp = startCircuit<CallGateOp>(moduleOp, circuitName, callGateOp);
    OpBuilder circuitBuilder =
        OpBuilder::atBlockBegin(&circOp.getBody().front());
    auto newCallGateOp =
        circuitBuilder.create<CallGateOp>(callGateOp.getLoc(), StringRef("x"),
                                          TypeRange{}, circOp.getArguments());
    finishCircuit(circOp);
    resetGateCircuitsSymbolMap[llvm::StringRef(circuitName)] =
        circOp.getOperation();
  } else
    circOp = dyn_cast<CircuitOp>(search->second);

  mlir::OpBuilder builder(callGateOp);
  auto callCircOp = builder.create<mlir::quir::CallCircuitOp>(
      callGateOp->getLoc(), circOp.getSymName(), TypeRange{},
      callGateOp.getOperands());

  callCircOp->moveBefore(callGateOp);
  callGateOp->erase();
}

void BreakResetPass::insertMeasureInCircuit(ModuleOp moduleOp,
                                            mlir::quir::MeasureOp measureOp) {
  const std::set<uint32_t> measureQubits =
      QubitOpInterface::getOperatedQubits(measureOp);
  ;
  std::string circuitName = "reset_measure";
  for (auto qubit : measureQubits)
    circuitName += ("_" + std::to_string(qubit));

  mlir::OpBuilder builder(measureOp);
  std::vector<mlir::Type> const typeVec(measureOp.getQubits().size(),
                                        builder.getI1Type());
  auto search = resetGateCircuitsSymbolMap.find(circuitName);
  CircuitOp circOp;
  if (search == resetGateCircuitsSymbolMap.end()) {
    // build a circuit
    circOp = startCircuit<MeasureOp>(moduleOp, circuitName, measureOp);
    OpBuilder circuitBuilder =
        OpBuilder::atBlockBegin(&circOp.getBody().front());
    auto resetMeasureOp = circuitBuilder.create<MeasureOp>(
        measureOp.getLoc(), TypeRange(typeVec), circOp.getArguments());
    resetMeasureOp->setAttr(getNoReportRuntimeAttrName(),
                            builder.getUnitAttr());
    finishCircuit(circOp);
    resetGateCircuitsSymbolMap[llvm::StringRef(circuitName)] =
        circOp.getOperation();
  } else
    circOp = dyn_cast<CircuitOp>(search->second);

  auto callCircOp = builder.create<mlir::quir::CallCircuitOp>(
      measureOp->getLoc(), circOp.getSymName(), TypeRange(typeVec),
      measureOp.getOperands());

  callCircOp->moveBefore(measureOp);
  measureOp->replaceAllUsesWith(callCircOp);

  measureOp->erase();
}

template <class measureOrCallGate>
CircuitOp BreakResetPass::startCircuit(ModuleOp moduleOp,
                                       const std::string &circuitName,
                                       measureOrCallGate quantumGate) {
  mlir::func::FuncOp mainFunc =
      dyn_cast<mlir::func::FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");
  mlir::OpBuilder builder(mainFunc);
  const mlir::Location location = mainFunc.getLoc();

  // mangle the circuit name if there exist a circuit with the same name in
  // input of this pass
  auto mangledCircuitName = circuitName;
  int cnt = 0;
  while (inputCircuitsSymbolMap.contains(mangledCircuitName))
    mangledCircuitName += std::to_string(cnt++);

  auto circOp = builder.create<CircuitOp>(mainFunc.getLoc(), mangledCircuitName,
                                          builder.getFunctionType(
                                              /*inputs=*/ArrayRef<Type>(),
                                              /*results=*/ArrayRef<Type>()));

  circOp.addEntryBlock();
  OpBuilder circuitBuilder = OpBuilder::atBlockBegin(&circOp.getBody().front());

  uint argumentIndex = 0;
  for (auto operand : quantumGate->getOperands()) {
    circOp.insertArgument(argumentIndex, operand.getType(), {},
                          operand.getLoc());
    argumentIndex++;
  }

  circuitBuilder.create<mlir::quir::ReturnOp>(location, ValueRange({}));
  return circOp;
}

void BreakResetPass::finishCircuit(mlir::quir::CircuitOp circOp) {
  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Type> inputTypes;
  llvm::SmallVector<Value> outputValues;
  OpBuilder circuitBuilder = OpBuilder::atBlockBegin(&circOp.getBody().front());

  bool measOpFound = false;
  circOp.walk([&](MeasureOp measOp) {
    measOpFound = true;
    inputTypes = TypeRange(measOp.getOperandTypes());
    outputTypes.append(measOp.result_type_begin(), measOp.result_type_end());
    outputValues.append(measOp.result_begin(), measOp.result_end());
    return WalkResult::interrupt();
  });

  bool callGateOpFound = false;
  circOp.walk([&](CallGateOp callGateOp) {
    callGateOpFound = true;
    inputTypes = TypeRange(callGateOp.getOperandTypes());
    outputTypes.append(callGateOp->result_type_begin(),
                       callGateOp->result_type_end());
    outputValues.append(callGateOp->result_begin(), callGateOp->result_end());
    return WalkResult::interrupt();
  });

  // sanity check that only one of measOp or callGateOp were found
  assert(
      ((callGateOpFound && !measOpFound) ||
       (!callGateOpFound && measOpFound)) &&
      "only one of measure op or call gate op can be present in input circuit");

  circOp.setType(circuitBuilder.getFunctionType(
      /*inputs=*/ArrayRef<Type>(inputTypes),
      /*results=*/ArrayRef<Type>(outputTypes)));

  circOp.walk([&](mlir::quir::ReturnOp returnOp) {
    returnOp->insertOperands(0, ValueRange(outputValues));
    return WalkResult::interrupt();
  });
}

llvm::StringRef BreakResetPass::getArgument() const { return "break-reset"; }
llvm::StringRef BreakResetPass::getDescription() const {
  return "Break reset ops into repeated measure and conditional x gate calls";
}

llvm::StringRef BreakResetPass::getName() const { return "Break Reset Pass"; }

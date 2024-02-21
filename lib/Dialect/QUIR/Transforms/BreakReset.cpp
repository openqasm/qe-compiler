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
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIREnums.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
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

      if (thisPass_.PUT_QUANTUM_GATES_INTO_CIRC)
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
        if (thisPass_.PUT_QUANTUM_GATES_INTO_CIRC)
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
  // check for command line override of PUT_QUANTUM_GATES_INTO_CIRC
  if (putCallGatesAndMeasuresIntoCircuit.hasValue())
    PUT_QUANTUM_GATES_INTO_CIRC = putCallGatesAndMeasuresIntoCircuit.getValue();

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

  // put measures and call gates into circuits -- when
  // putCallGatesAndMeasuresIntoCircuit option is true
  mlir::ModuleOp moduleOp = getOperation();
  uint circNum = 0;
  while (!measureList.empty()) {
    MeasureOp measOp = dyn_cast<MeasureOp>(measureList.front());
    putMeasureInCircuit(moduleOp, measOp, circNum);
    circNum++;
    measureList.pop_front();
  }
  while (!callGateList.empty()) {
    CallGateOp callGateOp = dyn_cast<CallGateOp>(callGateList.front());
    putCallGateInCircuit(moduleOp, callGateOp, circNum);
    circNum++;
    callGateList.pop_front();
  }
} // BreakResetPass::runOnOperation

void BreakResetPass::putCallGateInCircuit(ModuleOp moduleOp,
                                          mlir::quir::CallGateOp callGateOp,
                                          uint circNum) {
  mlir::func::FuncOp mainFunc =
      dyn_cast<mlir::func::FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");
  mlir::OpBuilder builder(mainFunc);
  mlir::Location location = mainFunc.getLoc();

  auto circOp = builder.create<CircuitOp>(
      mainFunc.getLoc(), "reset_x_" + std::to_string(circNum++),
      builder.getFunctionType(
          /*inputs=*/ArrayRef<Type>(),
          /*results=*/ArrayRef<Type>()));

  OpBuilder circuitBuilder = OpBuilder::atBlockBegin(&circOp.getBody().front());

  uint argumentIndex = 0;
  llvm::SmallVector<Type> inputTypes;
  llvm::SmallVector<Value> inputValues;
  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;

  for (auto operand : callGateOp->getOperands()) {
    inputTypes.push_back(operand.getType());
    inputValues.push_back(operand);
    circOp.insertArgument(argumentIndex, operand.getType(), {},
                          operand.getLoc());
    argumentIndex++;
  }

  auto newCallGateOp = circuitBuilder.create<CallGateOp>(
      location, StringRef("x"), TypeRange{}, circOp.getArguments());

  outputValues.append(newCallGateOp->result_begin(),
                      newCallGateOp->result_end());
  outputTypes.append(newCallGateOp->result_type_begin(),
                     newCallGateOp->result_type_end());

  circOp.setType(circuitBuilder.getFunctionType(
      /*inputs=*/ArrayRef<Type>(inputTypes),
      /*results=*/ArrayRef<Type>(outputTypes)));

  auto returnOp =
      circuitBuilder.create<mlir::quir::ReturnOp>(location, ValueRange({}));
  returnOp->insertOperands(0, ValueRange(outputValues));

  auto callCircOp = builder.create<mlir::quir::CallCircuitOp>(
      circOp->getLoc(), circOp.getName(), TypeRange(outputTypes),
      ValueRange(inputValues));

  callCircOp->moveBefore(callGateOp);
  callGateOp->erase();
}

void BreakResetPass::putMeasureInCircuit(ModuleOp moduleOp,
                                         mlir::quir::MeasureOp measureOp,
                                         uint circNum) {
  mlir::func::FuncOp mainFunc =
      dyn_cast<mlir::func::FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");
  mlir::OpBuilder builder(mainFunc);

  mlir::Location location = mainFunc.getLoc();
  auto circOp = builder.create<CircuitOp>(
      mainFunc.getLoc(), "reset_measure_" + std::to_string(circNum++),
      builder.getFunctionType(
          /*inputs=*/ArrayRef<Type>(),
          /*results=*/ArrayRef<Type>()));

  OpBuilder circuitBuilder = OpBuilder::atBlockBegin(&circOp.getBody().front());

  uint argumentIndex = 0;
  llvm::SmallVector<Type> inputTypes;
  llvm::SmallVector<Value> inputValues;
  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;

  for (auto operand : measureOp->getOperands()) {
    inputTypes.push_back(operand.getType());
    inputValues.push_back(operand);
    circOp.insertArgument(argumentIndex, operand.getType(), {},
                          operand.getLoc());
    argumentIndex++;
  }

  std::vector<mlir::Type> const typeVec(measureOp.getQubits().size(),
                                        circuitBuilder.getI1Type());

  auto resetMeasureOp = circuitBuilder.create<MeasureOp>(
      location, TypeRange(typeVec), circOp.getArguments());
  resetMeasureOp->setAttr(getNoReportRuntimeAttrName(), builder.getUnitAttr());

  outputValues.append(resetMeasureOp.result_begin(),
                      resetMeasureOp.result_end());
  outputTypes.append(resetMeasureOp.result_type_begin(),
                     resetMeasureOp.result_type_end());

  circOp.setType(circuitBuilder.getFunctionType(
      /*inputs=*/ArrayRef<Type>(inputTypes),
      /*results=*/ArrayRef<Type>(outputTypes)));

  auto returnOp =
      circuitBuilder.create<mlir::quir::ReturnOp>(location, ValueRange({}));
  returnOp->insertOperands(0, ValueRange(outputValues));

  auto callCircOp = builder.create<mlir::quir::CallCircuitOp>(
      circOp->getLoc(), circOp.getName(), TypeRange(outputTypes),
      ValueRange(inputValues));

  callCircOp->moveBefore(measureOp);
  measureOp->replaceAllUsesWith(callCircOp);

  measureOp->erase();
}

llvm::StringRef BreakResetPass::getArgument() const { return "break-reset"; }
llvm::StringRef BreakResetPass::getDescription() const {
  return "Break reset ops into repeated measure and conditional x gate calls";
}

llvm::StringRef BreakResetPass::getName() const { return "Break Reset Pass"; }

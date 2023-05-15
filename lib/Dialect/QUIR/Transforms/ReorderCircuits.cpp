//===- ReorderCircuits.cpp - Move call_circuits ops later -------*- C++ -*-===//
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
///  This file implements the pass for moving call_circuits as late as possible
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/ReorderCircuits.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/Support/Debug.h>

#include <algorithm>
#include <mlir/Dialect/Affine/IR/AffineOps.h>

#define DEBUG_TYPE "QUIRReorderMeasurements"

using namespace mlir;
using namespace mlir::quir;

namespace {
// This pattern matches on a measure op and a non-measure op and moves the
// non-measure op to occur earlier lexicographically if that does not change
// the topological ordering
struct ReorderCircuitsAndNonCircuitPat
    : public OpRewritePattern<CallCircuitOp> {
  explicit ReorderCircuitsAndNonCircuitPat(MLIRContext *ctx)
      : OpRewritePattern<CallCircuitOp>(ctx) {}

  LogicalResult matchAndRewrite(CallCircuitOp callCircuitOp,
                                PatternRewriter &rewriter) const override {

    // Accumulate qubits in measurement set
    std::set<uint> currQubits = callCircuitOp.getOperatedQubits();
    LLVM_DEBUG(llvm::dbgs() << "Matching on call_circuit for qubits:\t");
    LLVM_DEBUG(for (uint id : currQubits) llvm::dbgs() << id << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");

    auto nextAffineStoreOpp =
        dyn_cast<mlir::AffineStoreOp>(callCircuitOp->getNextNode());
    if (nextAffineStoreOpp) {
      bool moveAffine = true;
      for (auto operand : nextAffineStoreOpp->getOperands())
        if (operand.getDefiningOp() == callCircuitOp)
          moveAffine = false;
      if (moveAffine) {
        nextAffineStoreOpp->moveBefore(callCircuitOp);
        return success();
      }
    }

    return failure();
  } // matchAndRewrite
};  // struct ReorderCircuitsAndNonCircuitPat
} // anonymous namespace

void ReorderCircuitsPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<ReorderCircuitsAndNonCircuitPat>(&getContext());

  FuncOp mainFunc = dyn_cast<FuncOp>(getMainFunction(moduleOperation));

  if (!mainFunc) {
    signalPassFailure();
    llvm::errs() << "Unable to find main function! Cannot add shot loop\n";
    return;
  }

  // only run this pass on call_circuits within the main body of the program
  // there may be call_circuits within circuits that have not been properly
  // labeled with their qubit arguments

  if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns))))
    signalPassFailure();
} // runOnOperation

llvm::StringRef ReorderCircuitsPass::getArgument() const {
  return "reorder-circuits";
}

llvm::StringRef ReorderCircuitsPass::getDescription() const {
  return "Move call_circuits to be as lexicograpically as late as possible "
         "without "
         "affecting the topological ordering.";
}

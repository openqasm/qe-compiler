//===- ReorderMeasurements.cpp - Move measurement ops later -----*- C++ -*-===//
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
///  This file implements the pass for moving measurements as late as possible
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/ReorderMeasurements.h"

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/Support/Debug.h>

#include <algorithm>

#define DEBUG_TYPE "QUIRReorderMeasurements"

using namespace mlir;
using namespace mlir::quir;

namespace {
// This pattern matches on a measure op and a non-measure op and moves the
// non-measure op to occur earlier lexicographically if that does not change
// the topological ordering
struct ReorderMeasureAndNonMeasurePat : public OpRewritePattern<MeasureOp> {
  explicit ReorderMeasureAndNonMeasurePat(MLIRContext *ctx)
      : OpRewritePattern<MeasureOp>(ctx) {}

  LogicalResult matchAndRewrite(MeasureOp measureOp,
                                PatternRewriter &rewriter) const override {
    bool anyMove = false;
    do {
      // Accumulate qubits in measurement set
      std::set<uint> currQubits = measureOp.getOperatedQubits();
      LLVM_DEBUG(llvm::dbgs() << "Matching on measurement for qubits:\t");
      LLVM_DEBUG(for (uint id : currQubits) llvm::dbgs() << id << " ");
      LLVM_DEBUG(llvm::dbgs() << "\n");

      auto nextOpt = nextQuantumOrControlFlowOrNull(measureOp);
      if (!nextOpt.hasValue())
        break;

      Operation *nextOp = nextOpt.getValue();
      // for control flow ops, continue, but add the operated qubits of the
      // control flow block to the currQubits set
      while (nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
        addQubitIdsFromAttr(nextOp, currQubits);

        // now find the next next op
        auto nextNextOpt = nextQuantumOrControlFlowOrNull(nextOp);
        if (!nextNextOpt.hasValue()) // only move non-control-flow ops
          break;

        nextOp = nextNextOpt.getValue();
      }

      // don't reorder past the next measurement or reset or control flow
      if (nextOp->hasTrait<mlir::quir::CPTPOp>() ||
          nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>())
        break;

      // Check for overlap between currQubits and what's operated on by nextOp
      std::set<uint> nextQubits = QubitOpInterface::getOperatedQubits(nextOp);
      if (QubitOpInterface::qubitSetsOverlap(currQubits, nextQubits))
        break;

      // Make sure that the nextOp doesn't use an SSA value defined between
      // the measureOp and nextOp
      Block *measBlock = measureOp->getBlock();
      bool interveningValue = false;
      for (auto operand : nextOp->getOperands())
        if (Operation *defOp = operand.getDefiningOp())
          if (defOp->getBlock() == measBlock &&
              measureOp->isBeforeInBlock(defOp)) {
            interveningValue = true;
            break;
          }

      if (interveningValue)
        break;

      LLVM_DEBUG(llvm::dbgs() << "Succeeded match with operation:\n");
      LLVM_DEBUG(nextOp->dump());
      LLVM_DEBUG(llvm::dbgs() << "on qubits:\t");
      LLVM_DEBUG(for (uint id // this is ugly but clang-format insists
                      : QubitOpInterface::getOperatedQubits(nextOp)) {
        llvm::dbgs() << id << " ";
      });
      LLVM_DEBUG(llvm::dbgs() << "\n\n");

      // good to move the nextOp before the measureOp
      nextOp->moveBefore(measureOp);
      anyMove = true;
    } while (true);

    if (anyMove)
      return success();

    return failure();
  } // matchAndRewrite
};  // struct ReorderMeasureAndNonMeasurePat
} // anonymous namespace

void ReorderMeasurementsPass::runOnOperation() {
  Operation *moduleOperation = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.insert<ReorderMeasureAndNonMeasurePat>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns))))
    signalPassFailure();
} // runOnOperation

llvm::StringRef ReorderMeasurementsPass::getArgument() const {
  return "reorder-measures";
}

llvm::StringRef ReorderMeasurementsPass::getDescription() const {
  return "Move qubits to be as lexicograpically as late as possible without "
         "affecting the topological ordering.";
}

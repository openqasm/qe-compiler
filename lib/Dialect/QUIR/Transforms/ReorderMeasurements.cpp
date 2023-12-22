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

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <iterator>
#include <vector>

#define DEBUG_TYPE "QUIRReorderMeasurements"

using namespace mlir;
using namespace mlir::quir;

namespace {

using MoveListVec = std::vector<Operation *>;

bool mayMoveVariableLoadOp(MeasureOp measureOp,
                           oq3::VariableLoadOp variableLoadOp,
                           MoveListVec &moveList);
bool mayMoveCastOp(MeasureOp measureOp, oq3::CastOp castOp,
                   MoveListVec &moveList);

bool mayMoveVariableLoadOp(MeasureOp measureOp,
                           oq3::VariableLoadOp variableLoadOp,
                           MoveListVec &moveList) {
  // find corresponding variable assign
  // move variableLoad if the assign is before the measure
  bool moveVariableLoadOp = true;
  auto *currentBlock = variableLoadOp->getBlock();
  currentBlock->walk([&](oq3::VariableAssignOp assignOp) {
    if (assignOp.getVariableName() == variableLoadOp.getVariableName()) {
      moveVariableLoadOp = assignOp->isBeforeInBlock(measureOp);
      if (!moveVariableLoadOp) {
        auto assignCastOp =
            dyn_cast<oq3::CastOp>(assignOp.getAssignedValue().getDefiningOp());
        if (assignCastOp)
          moveVariableLoadOp = mayMoveCastOp(measureOp, assignCastOp, moveList);
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (moveVariableLoadOp)
    moveList.push_back(variableLoadOp);
  return moveVariableLoadOp;
}

bool mayMoveCastOp(MeasureOp measureOp, oq3::CastOp castOp,
                   MoveListVec &moveList) {
  bool moveCastOp = false;
  auto variableLoadOp =
      dyn_cast<oq3::VariableLoadOp>(castOp.getArg().getDefiningOp());
  if (variableLoadOp)
    moveCastOp = mayMoveVariableLoadOp(measureOp, variableLoadOp, moveList);
  auto castMeasureOp = dyn_cast<MeasureOp>(castOp.getArg().getDefiningOp());
  if (castMeasureOp)
    moveCastOp = ((castMeasureOp != measureOp) &&
                  (castMeasureOp->isBeforeInBlock(measureOp) ||
                   castMeasureOp->getBlock() != castOp->getBlock()));

  if (moveCastOp)
    moveList.push_back(castOp);

  return moveCastOp;
}

// This pattern matches on a measure op and a non-measure op and moves the
// non-measure op to occur earlier lexicographically if that does not change
// the topological ordering
struct ReorderMeasureAndNonMeasurePat : public OpRewritePattern<MeasureOp> {
  explicit ReorderMeasureAndNonMeasurePat(MLIRContext *ctx)
      : OpRewritePattern<MeasureOp>(ctx) {}

  LogicalResult matchAndRewrite(MeasureOp measureOp,
                                PatternRewriter &rewriter) const override {
    bool anyMove = false;

    MoveListVec moveList;

    do {
      // Accumulate qubits in measurement set
      std::set<uint> currQubits = measureOp.getOperatedQubits();
      LLVM_DEBUG(llvm::dbgs() << "Matching on measurement for qubits:\t");
      LLVM_DEBUG(for (uint id : currQubits) llvm::dbgs() << id << " ");
      LLVM_DEBUG(llvm::dbgs() << "\n");

      auto nextOpt = nextQuantumOrControlFlowOrNull(measureOp);
      if (!nextOpt.has_value())
        break;

      Operation *nextOp = nextOpt.value();
      // for control flow ops, continue, but add the operated qubits of the
      // control flow block to the currQubits set
      while (nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
        addQubitIdsFromAttr(nextOp, currQubits);

        // now find the next next op
        auto nextNextOpt = nextQuantumOrControlFlowOrNull(nextOp);
        if (!nextNextOpt.has_value()) // only move non-control-flow ops
          break;

        nextOp = nextNextOpt.value();
      }

      // don't reorder past the next measurement or reset or control flow
      if (nextOp->hasTrait<mlir::quir::CPTPOp>() ||
          nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>())
        break;

      // Check for overlap between currQubits and what's operated on by nextOp
      std::set<uint> nextQubits = QubitOpInterface::getOperatedQubits(nextOp);
      if (QubitOpInterface::qubitSetsOverlap(currQubits, nextQubits))
        break;

      moveList.clear();

      // Make sure that the nextOp doesn't use an SSA value defined between
      // the measureOp and nextOp
      Block *measBlock = measureOp->getBlock();
      bool interveningValue = false;
      for (auto operand : nextOp->getOperands())
        if (Operation *defOp = operand.getDefiningOp())
          if (defOp->getBlock() == measBlock &&
              measureOp->isBeforeInBlock(defOp)) {

            bool moveOps = false;

            // if the defining op is a variable load attempt to move it above
            // the measurement
            auto variableLoadOp = dyn_cast<oq3::VariableLoadOp>(defOp);
            if (variableLoadOp) {
              moveOps =
                  mayMoveVariableLoadOp(measureOp, variableLoadOp, moveList);
            }

            auto castOp = dyn_cast<oq3::CastOp>(defOp);
            if (castOp)
              moveOps = mayMoveCastOp(measureOp, castOp, moveList);

            if (moveOps) {
              Operation *mbOp = measureOp.getOperation();
              for (auto op = moveList.rbegin(); op != moveList.rend(); ++op) {
                (*op)->moveBefore(mbOp);
                mbOp = *op;
              }
              continue;
            }

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
  patterns.add<ReorderMeasureAndNonMeasurePat>(&getContext());

  mlir::GreedyRewriteConfig config;
  // Disable to improve performance
  config.enableRegionSimplification = false;

  if (failed(applyPatternsAndFoldGreedily(moduleOperation, std::move(patterns),
                                          config)))
    signalPassFailure();
} // runOnOperation

llvm::StringRef ReorderMeasurementsPass::getArgument() const {
  return "reorder-measures";
}

llvm::StringRef ReorderMeasurementsPass::getDescription() const {
  return "Move qubits to be as lexicograpically as late as possible without "
         "affecting the topological ordering.";
}

//===- AddShotLoop.cpp - Add shot loop --------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the pass for adding a for shot loop around the entire
//  main function body
//
//===----------------------------------------------------------------------===//

#include <list>

#include "Dialect/QUIR/Transforms/AddShotLoop.h"

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"
#include "Dialect/QuSys/IR/QuSysAttributes.h"
#include "Dialect/QuSys/IR/QuSysOps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::quir;
using namespace mlir::qusys;

// Entry point for the pass.
void AddShotLoopPass::runOnOperation() {
  // This pass is only called on module Ops
  ModuleOp moduleOp = getOperation();
  FuncOp mainFunc = dyn_cast<FuncOp>(getMainFunction(moduleOp));

  if (!mainFunc) {
    signalPassFailure();
    llvm::errs() << "Unable to find main function! Cannot add shot loop\n";
    return;
  }

  // start the builder outside the main function so we aren't cloning or
  // building into the same region that we are copying from
  OpBuilder build(moduleOp.body());
  Location opLoc = mainFunc.getLoc();

  auto startOp = build.create<mlir::arith::ConstantOp>(
      opLoc, build.getIndexType(), build.getIndexAttr(0));
  auto endOp = build.create<mlir::arith::ConstantOp>(
      opLoc, build.getIndexType(), build.getIndexAttr(numShots));
  auto stepOp = build.create<mlir::arith::ConstantOp>(
      opLoc, build.getIndexType(), build.getIndexAttr(1));
  auto forOp = build.create<scf::ForOp>(opLoc, startOp, endOp, stepOp);
  forOp->setAttr(getShotLoopAttrName(), build.getUnitAttr());

  build.setInsertionPointToStart(&forOp.getRegion().front());
  auto shotInit = build.create<ShotInitOp>(opLoc);
  shotInit->setAttr(getNumShotsAttrName(), build.getI32IntegerAttr(numShots));

  std::list<Operation *> toErase;
  BlockAndValueMapping mapper;

  Operation *lastOp = nullptr;
  for (Operation &op : mainFunc.body().getOps()) {
    if (dyn_cast<SystemInitOp>(&op))
      continue;
    if (dyn_cast<SystemFinalizeOp>(&op) || dyn_cast<mlir::ReturnOp>(&op)) {
      lastOp = &op;
      break;
    }
    build.clone(op, mapper);
    toErase.emplace_front(&op); // must erase in reverse order
  }

  for (auto *operation : toErase)
    if (operation->use_empty())
      operation->erase();

  if (!lastOp) {
    // if last op wasn't detected while iterating
    // set it to the last op in the one block of the mainFunc body region
    lastOp = &mainFunc.body().front().back();
  }

  startOp->moveBefore(lastOp);
  endOp->moveBefore(lastOp);
  stepOp->moveBefore(lastOp);
  forOp->moveBefore(lastOp);
} // runOnOperation

llvm::StringRef AddShotLoopPass::getArgument() const { return "add-shot-loop"; }
llvm::StringRef AddShotLoopPass::getDescription() const {
  return "Add a for loop wrapping the main function body iterating over the "
         "number of shots";
}

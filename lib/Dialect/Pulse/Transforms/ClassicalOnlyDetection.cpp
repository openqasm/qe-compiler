//===- ClassicalOnlyDetection.cpp - detect pulse ops ------------*- C++ -*-===//
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
/// Defines pass for updating quir.classicalOnly flag based on the presence of
/// Pulse dialect Ops
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/ClassicalOnlyDetection.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTypes.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <llvm/Support/Casting.h>

using namespace mlir;
using namespace mlir::pulse;

namespace mlir::pulse {

// detects whether or not an operation contains pulse operations inside
auto ClassicalOnlyDetectionPass::hasPulseSubOps(Operation *inOp) -> bool {
  bool retVal = false;
  inOp->walk([&](Operation *op) {
    if (llvm::isa<pulse::PulseDialect>(op->getDialect())) {
      retVal = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return retVal;
} // ClassicalOnlyDetectionPass::hasPulseSubOps

// Entry point for the ClassicalOnlyDetectionPass pass
void ClassicalOnlyDetectionPass::runOnOperation() {
  // This pass is only called on the top-level module Op
  Operation *moduleOperation = getOperation();
  OpBuilder b(moduleOperation);

  moduleOperation->walk([&](Operation *op) {
    if (dyn_cast<scf::IfOp>(op) || dyn_cast<scf::ForOp>(op) ||
        dyn_cast<quir::SwitchOp>(op) || dyn_cast<SequenceOp>(op) ||
        dyn_cast<mlir::func::FuncOp>(op)) {
      // check for a pre-existing classicalOnly attribute
      // only update if the attribute does not exist or it is true
      // indicating that no quantum ops have been identified yet
      auto attrName = llvm::StringRef("quir.classicalOnly");
      auto classicalOnlyAttr = op->getAttrOfType<BoolAttr>(attrName);
      if (!classicalOnlyAttr || classicalOnlyAttr.getValue())
        op->setAttr(attrName, b.getBoolAttr(!hasPulseSubOps(op)));
    }
  });
} // ClassicalOnlyDetectionPass::runOnOperation

llvm::StringRef ClassicalOnlyDetectionPass::getArgument() const {
  return "pulse-classical-only-detection";
}
llvm::StringRef ClassicalOnlyDetectionPass::getDescription() const {
  return "Detect control flow blocks that contain only classical (non-quantum) "
         "operations, and decorate them with a classicalOnly bool attribute";
}
} // namespace mlir::pulse

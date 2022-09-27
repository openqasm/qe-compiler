//===- Analysis.cpp - QUIR Analyses -----------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares analyses QUIR
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/Analysis.h"

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/SCF/SCF.h"

namespace mlir::quir {

PurelyUnitaryAnalysis::PurelyUnitaryAnalysis(mlir::Operation *op) {
  mlir::WalkResult result = op->walk([&](mlir::Operation *op) {
    if (op->hasTrait<mlir::quir::UnitaryOp>() or
        llvm::isa<mlir::scf::YieldOp>(op))
      return mlir::WalkResult::advance();
    return WalkResult::interrupt();
  });

  if (result.wasInterrupted())
    isPurelyUnitary = false;
}

} // namespace mlir::quir

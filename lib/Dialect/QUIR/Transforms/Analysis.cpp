//===- Analysis.cpp - QUIR Analyses -----------------------------*- C++ -*-===//
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

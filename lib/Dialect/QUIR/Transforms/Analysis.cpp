//===- Analysis.cpp - QUIR Analyses -----------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

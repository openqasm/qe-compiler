//===- Analysis.h - QUIR Analyses -------------------------------*- C++ -*-===//
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
///  This file declares QUIR analyses.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_ANALYSIS_H
#define QUIR_ANALYSIS_H

#include "mlir/IR/Operation.h"

namespace mlir::quir {

// Validate that all operations contained within an operation are unitary
// or a scf::yieldOp operation.
// TODO: This should be replace by analysis performed on a canonicalized
// quir.circuit operation
struct PurelyUnitaryAnalysis {

  bool isPurelyUnitary = true;

  PurelyUnitaryAnalysis(mlir::Operation *op);
};

} // namespace mlir::quir

#endif // QUIR_ANALYSIS_H

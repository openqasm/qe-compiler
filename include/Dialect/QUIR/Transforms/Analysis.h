//===- Analysis.h - QUIR Analyses -------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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

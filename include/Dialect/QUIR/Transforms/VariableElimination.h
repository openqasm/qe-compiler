//===- VariableElimination.h - Lower and eliminate variables ----*- C++ -*-===//
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
///  This file declares the passes for converting QUIR variables to memref
///  operations and eliminating them where possible.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_VARIABLE_ELIMINATION_H
#define QUIR_VARIABLE_ELIMINATION_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::quir {

struct VariableEliminationPass
    : public PassWrapper<VariableEliminationPass, OperationPass<>> {
  void runOnOperation() override;

  bool externalizeOutputVariables;

  VariableEliminationPass(bool externalizeOutputVariables = false)
      : PassWrapper(), externalizeOutputVariables(externalizeOutputVariables) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::AffineDialect>();
  }
}; // struct VariableEliminationPass

} // namespace mlir::quir

#endif // QUIR_VARIABLE_ELIMINATION_H

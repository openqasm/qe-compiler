//===- ReorderCircuits.h - Move call_circuits ops later ---------*- C++ -*-===//
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
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for moving call_circuits operations later
///  when possible.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_REORDER_CIRCUITS_H
#define QUIR_REORDER_CIRCUITS_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

/// @brief Move calll_circuits when possible
struct ReorderCircuitsPass
    : public PassWrapper<ReorderCircuitsPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct ReorderCircuitsPass

} // namespace mlir::quir

#endif // QUIR_REORDER_CIRCUITS_H

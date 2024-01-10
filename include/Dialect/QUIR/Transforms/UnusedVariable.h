//===- UnusedVariable.h - Remove unused variables ---------------*- C++ -*-===//
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
///  This file declares the pass for removing variables that are unused
///  by any subsequent load
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_UNUSED_VARIABLE_H
#define QUIR_UNUSED_VARIABLE_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

///
/// \brief Remove unused variables
/// \details This pass removes QUIR variables that are not followed by
/// a subsequent variable load/use, unless they are declared with the 'output'
/// attribute.
struct UnusedVariablePass
    : public PassWrapper<UnusedVariablePass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
}; // struct UnusedVariablePass
} // namespace mlir::quir

#endif // QUIR_UNUSED_VARIABLE_H

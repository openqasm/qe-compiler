//===- UnusedVariable.h - Remove unused variables ---------------*- C++ -*-===//
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
}; // struct UnusedVariablePass
} // namespace mlir::quir

#endif // QUIR_UNUSED_VARIABLE_H

//===- QUIRToStd.h - Convert QUIR to Std Dialect ----------------*- C++ -*-===//
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
//
//  This file declares the pass for converting QUIR to std dialect
//
//===----------------------------------------------------------------------===//

#ifndef MOCK_CONVERSION_QUIRTOSTD_H
#define MOCK_CONVERSION_QUIRTOSTD_H

#include "MockTarget.h"

#include "HAL/TargetOperationPass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace qssc::targets::mock::conversion {
struct MockQUIRToStdPass
    : public mlir::PassWrapper<
          MockQUIRToStdPass,
          hal::TargetOperationPass<MockSystem, mlir::ModuleOp>> {
  void runOnOperation(MockSystem &system) override;
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  bool externalizeOutputVariables;

  MockQUIRToStdPass(bool externalizeOutputVariables)
      : PassWrapper(), externalizeOutputVariables(externalizeOutputVariables) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};
} // namespace qssc::targets::mock::conversion

#endif // MOCK_CONVERSION_QUIRTOSTD_H

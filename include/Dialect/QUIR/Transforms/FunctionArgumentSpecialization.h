//===- FunctionArgumentSpecialization.h - Resolve funcs ----------*- C++-*-===//
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
//  This file declares the pass for specializing function argument widths
//  to match calls. Specialized function defs are cloned with updated argument
//  widths
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_FUNCTION_ARGUMENT_SPECIALIZATION_H
#define QUIR_FUNCTION_ARGUMENT_SPECIALIZATION_H

#include "Dialect/QUIR/IR/QUIROps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

#include <deque>

namespace mlir::quir {
struct FunctionArgumentSpecializationPass
    : public PassWrapper<FunctionArgumentSpecializationPass, OperationPass<>> {

  template <class CallOpTy>
  void processCallOp(Operation *op, std::deque<Operation *> &callWorkList);

  template <class T1, class T2, class... Rest>
  void processCallOp(Operation *op, std::deque<Operation *> &callWorkList);

  template <class CallOpTy>
  void copyFuncAndSpecialize(FuncOp inFunc, CallOpTy callOp,
                             std::deque<Operation *> &callWorkList);

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct FunctionArgumentSpecializationPass
} // namespace mlir::quir

#endif // QUIR_FUNCTION_ARGUMENT_SPECIALIZATION_H

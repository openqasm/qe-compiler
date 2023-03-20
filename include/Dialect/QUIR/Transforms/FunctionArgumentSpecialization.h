//===- FunctionArgumentSpecialization.h - Resolve funcs ----------*- C++-*-===//
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

//===- Errors.h -------------------------------------------------*- C++ -*-===//
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
//  This file declares the classes for errors of aer-simulator
//
//===----------------------------------------------------------------------===//
#ifndef TARGETS_SIMULATORS_AER_ERRORS_H
#define TARGETS_SIMULATORS_AER_ERRORS_H

#include <llvm/Support/Error.h>

namespace qssc::targets::simulators::aer {

class LLVMBuildFailure : public llvm::ErrorInfo<LLVMBuildFailure> {
public:
  static char ID;

  LLVMBuildFailure() = default;

  void log(llvm::raw_ostream &os) const override {
    os << "Failed to generate a binary file for Aer simulator";
  }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

} // namespace qssc::targets::simulators::aer

#endif // TARGETS_SIMULATORS_AER_ERRORS_H

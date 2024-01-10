//===- QUIRTestInterfaces.h - QUIR Dialect interface tests -*- C++ -*-========//
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
///  Tests for the QUIR dialect interfaces
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_TESTQUIRINTERFACES_H
#define QUIR_TESTQUIRINTERFACES_H

#include "mlir/Pass/Pass.h"

namespace mlir::quir {

//===----------------------------------------------------------------------===//
// QubitOpInterface
//===----------------------------------------------------------------------===//

struct TestQubitOpInterfacePass
    : public PassWrapper<TestQubitOpInterfacePass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
};

} // namespace mlir::quir

#endif // QUIR_QUIRTESTINTERFACES_H

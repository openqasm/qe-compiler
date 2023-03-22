//===- QUIRTestInterfaces.cpp - Test QUIR dialect interfaces ----*- C++ -*-===//
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
///  This file declares the QUIR dialect test interfaces in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRTestInterfaces.h"

#include "Dialect/QUIR/IR/QUIRInterfaces.h"

#include "mlir/IR/Builders.h"

using namespace mlir::quir;

//===----------------------------------------------------------------------===//
// QubitOpInterface
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace mlir::quir;

// Test annotation of used qubits for interface
void TestQubitOpInterfacePass::runOnOperation() {

  auto *op = getOperation();
  OpBuilder build(op);

  // Annotate all operations with used qubits
  op->walk([&](mlir::Operation *op) {
    auto opQubits = QubitOpInterface::getOperatedQubits(op);
    std::vector<int> vecOpQubits(opQubits.begin(), opQubits.end());
    std::sort(vecOpQubits.begin(), vecOpQubits.end());
    op->setAttr("quir.operatedQubits",
                build.getI32ArrayAttr(ArrayRef<int>(vecOpQubits)));
  });
}

llvm::StringRef TestQubitOpInterfacePass::getArgument() const {
  return "test-qubit-op-interface";
}
llvm::StringRef TestQubitOpInterfacePass::getDescription() const {
  return "Test QubitOpInterface by attributing operations with operated "
         "qubits.";
}

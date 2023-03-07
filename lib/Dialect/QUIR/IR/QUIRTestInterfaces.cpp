//===- QUIRTestInterfaces.cpp - Test QUIR dialect interfaces ----*- C++ -*-===//
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

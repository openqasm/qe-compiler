OPENQASM 3.0;
// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s

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

// CHECK: Manifest
// CHECK: MockAcquire_0.mlir
// CHECK: MockController.mlir
// CHECK: MockDrive_0.mlir
// CHECK: MockDrive_1.mlir
// CHECK: controller.bin
// CHECK: llvmModule.ll
qubit $0;
qubit $1;

gate cx control, target { }

bit c0;
bit c1;

U(1.57079632679, 0.0, 3.14159265359) $0;
cx $0, $1;
measure $0 -> c0;
measure $1 -> c1;

OPENQASM 3.0;
// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s

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

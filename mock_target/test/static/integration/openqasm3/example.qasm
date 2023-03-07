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

qubit $0;
bit c0;
U(1.57079632679, 0.0, 3.14159265359) $0;
measure $0 -> c0;

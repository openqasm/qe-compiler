OPENQASM 3.0;
// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload --enable-circuits=false| FileCheck %s
// RUN: cat %s | qss-compiler --include-source -X=qasm --target mock --config %TEST_CFG --emit=qem --plaintext-payload --enable-circuits=false | FileCheck %s --match-full-lines --check-prefix CHECK-SOURCE

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

qubit $0;
bit c0;
U(1.57079632679, 0.0, 3.14159265359) $0;
measure $0 -> c0;

// CHECK-SOURCE: manifest/input.qasm
// CHECK-SOURCE: qubit $0;
// CHECK-SOURCE: measure $0 -> c0;

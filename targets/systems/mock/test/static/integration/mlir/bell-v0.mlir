// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s
// RUN: cat %s | qss-compiler --include-source -X mlir --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s --match-full-lines --check-prefix CHECK-SOURCE
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
func.func @main () -> i32 {

  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %a0 = quir.constant #quir.angle<1.57079632679> : !quir.angle<20>
  %a1 = quir.constant #quir.angle<0.0> : !quir.angle<20>
  %a2 = quir.constant #quir.angle<3.14159265359> : !quir.angle<20>
  quir.builtin_U %q0, %a0, %a1, %a2 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
  quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// CHECK-SOURCE: File: manifest/input.mlir

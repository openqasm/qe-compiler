// RUN: qss-compiler -X=mlir %s | FileCheck %s

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.


// bit c0;
// bit c1;
quir.declare_variable @c0 : !quir.cbit<1>
quir.declare_variable @c1 : !quir.cbit<1>

func @main () -> i32 {

  // qubit %0;
  // qubit %1;
  // CHECK: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  // U(1.57079632679, 0.0, 3.14159265359) %0;
  %a0 = quir.constant #quir.angle<1.57079632679 : !quir.angle<20>>
  %a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
  %a2 = quir.constant #quir.angle<3.14159265359 : !quir.angle<20>>
  quir.builtin_U %q0, %a0, %a1, %a2 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
  // CX %0, %1;
  quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
  // measure %0 -> c0;
  // measure %1 -> c1;
  %mres1 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
  quir.assign_cbit_bit @c0<1> [0] : i1 = %mres1
  %mres2 = "quir.measure"(%q1) : (!quir.qubit<1>) -> i1
  quir.assign_cbit_bit @c1<1> [0] : i1 = %mres2
  %zero = arith.constant 0 : i32
  return %zero : i32
}

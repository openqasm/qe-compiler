// RUN: qss-compiler --canonicalize --quantum-decorate --reorder-measures %s | FileCheck %s

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

// This regression test case validates that reorder-measures does not move
// operations across control flow with other quantum operations.

// CHECK: module
module {
  oq3.declare_variable @b : !quir.cbit<1>
  oq3.declare_variable @results : !quir.cbit<1>
  func.func @x(%arg0: !quir.qubit<1>) {
    return
  }
  func.func @sx(%arg0: !quir.qubit<1>) {
    return
  }
  func.func @rz(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) {
    return
  }
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %angle = quir.constant #quir.angle<1.500000e+00> : !quir.angle<64>

    // CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32}
    // CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32}
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

    // CHECK: quir.measure([[QUBIT0]])
    %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1

    %6 = oq3.variable_load @b : !quir.cbit<1>
    %7 = "oq3.cast"(%6) : (!quir.cbit<1>) -> i1
    // CHECK: scf.if
    scf.if %7 {
      // CHECK: quir.call_gate @x([[QUBIT0]])
      quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
      // CHECK: quir.call_gate @rz([[QUBIT1]], {{.*}})
      quir.call_gate @rz(%1, %angle) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    } {quir.classicalOnly = false, quir.physicalIds = [0 : i32, 1 : i32]}

    // CHECK: quir.call_gate @sx([[QUBIT1]])
    quir.call_gate @sx(%1) : (!quir.qubit<1>) -> ()

    // CHECK: quir.measure([[QUBIT0]])
    %8 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    %9 = "oq3.cast"(%8) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @results : !quir.cbit<1> = %9
    return %c0_i32 : i32
  }
}

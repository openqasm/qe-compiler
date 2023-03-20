// RUN: qss-compiler -X=mlir %s --quantum-decorate | FileCheck %s

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


func @t1 (%cond : i1) -> () {
  %q0 = quir.declare_qubit {id = 0: i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1: i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2: i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3: i32} : !quir.qubit<1>
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32]}
  }
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32]}
  }
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32, 2 : i32, 3 : i32]}
  }
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    %res = "quir.measure"(%q1) : (!quir.qubit<1>) -> i1
    quir.reset %q0 : !quir.qubit<1>
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32, 2 : i32, 3 : i32]}
  }
  return
}

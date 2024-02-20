// RUN: qss-compiler -X=mlir --canonicalize --remove-unused-circuits %s | FileCheck %s
//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

module {
  // CHECK: quir.circuit @circuit_0(%arg0: !quir.qubit<1>) -> i1 {
  quir.circuit @circuit_0(%arg0: !quir.qubit<1> ) -> i1 {
    quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  // CHECK: quir.circuit @circuit_1(%arg0: !quir.qubit<1>) -> i1 {
  quir.circuit @circuit_1(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  // CHECK-NOT: quir.circuit @circuit_2(%arg0: !quir.qubit<1> ) -> i1 {
  quir.circuit @circuit_2(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  // CHECK-NOT: quir.circuit @circuit_3(%arg0: !quir.qubit<1> ) -> i1 {
  quir.circuit @circuit_3(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  func.func @main() -> i32  {
    %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

    %res1 = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> (i1)

    %res2 = quir.call_circuit @circuit_1(%q1) : (!quir.qubit<1>) -> (i1)

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

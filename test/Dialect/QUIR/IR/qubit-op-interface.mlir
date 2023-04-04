// RUN: qss-compiler -X=mlir --test-qubit-op-interface %s | FileCheck %s

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

// TODO: A separate attribute ID should not be required to identify qubit IDs
// This should be computable from the quir.call_circuit invocation. This would
// also mean that we would only have to declare a qubit once.
quir.circuit @circuit0 (%q0: !quir.qubit<1> {quir.physicalId = 0 : i32}) -> (i1, i1) {
// CHECK: quir.circuit @circuit0(%arg0: !quir.qubit<1> {quir.physicalId = 0 : i32}) -> (i1, i1) attributes {quir.operatedQubits = [0 : i32]} {
	quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0, %res0 : i1, i1
}

func @test_qubit_op_interface (%cond : i1) -> () {
  %q0 = quir.declare_qubit {id = 0: i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1: i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2: i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3: i32} : !quir.qubit<1>


  %res0, %res1 = quir.call_circuit @circuit0 (%q1) : (!quir.qubit<1>) -> (i1, i1)

  quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
  // CHECK: quir.call_gate @x({{.*}}) {quir.operatedQubits = [1 : i32]} : (!quir.qubit<1>) -> ()

  quir.barrier %q0, %q1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
  // CHECK: quir.barrier {{.*}}, {{.*}} {quir.operatedQubits = [0 : i32, 1 : i32]} : (!quir.qubit<1>, !quir.qubit<1>) -> ()

  %a0 = quir.constant #quir.angle<1.57079632679 : !quir.angle<20>>
  %a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
  %a2 = quir.constant #quir.angle<3.14159265359 : !quir.angle<20>>
  quir.builtin_U %q0, %a0, %a1, %a2 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
  // CHECK: quir.builtin_U {quir.operatedQubits = [0 : i32]} {{.*}}, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>

  quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
  // CHECK: quir.builtin_CX {quir.operatedQubits = [0 : i32, 1 : i32]} {{.*}}, {{.*}} : !quir.qubit<1>, !quir.qubit<1>

  %duration = quir.constant #quir.duration<"20ns" : !quir.duration>
  quir.delay %duration, (%q0) : !quir.duration, (!quir.qubit<1>) -> ()
  // CHECK: quir.delay {quir.operatedQubits = [0 : i32]} {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()

  qcs.delay_cycles (%q0) {time = 1 : i64} : (!quir.qubit<1>) -> ()
  // CHECK: qcs.delay_cycles({{.*}}) {quir.operatedQubits = [0 : i32], time = 1 : i64} : (!quir.qubit<1>) -> ()


  quir.reset %q0 : !quir.qubit<1>
  // CHECK: quir.reset {quir.operatedQubits = [0 : i32]} {{.*}} : !quir.qubit<1>

  %res2 = quir.measure (%q0) : (!quir.qubit<1>) -> i1
  // CHECK: {{.*}} = quir.measure({{.*}}) {quir.operatedQubits = [0 : i32]} : (!quir.qubit<1>) -> i1

  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    // CHECK: quir.call_gate @x({{.*}}) {quir.operatedQubits = [0 : i32]} : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    // CHECK: quir.call_gate @x({{.*}}) {quir.operatedQubits = [2 : i32]} : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    // CHECK: quir.call_gate @x({{.*}}) {quir.operatedQubits = [1 : i32]} : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    // CHECK: quir.call_gate @x({{.*}}) {quir.operatedQubits = [3 : i32]} : (!quir.qubit<1>) -> ()
  }
  // CHECK: } {quir.operatedQubits = [0 : i32, 1 : i32, 2 : i32, 3 : i32]}

  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    %res = "quir.measure"(%q1) : (!quir.qubit<1>) -> i1
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    // CHECK: quir.call_gate @x({{.*}}) {quir.operatedQubits = [3 : i32]} : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    // CHECK: quir.call_gate @x({{.*}}) {quir.operatedQubits = [2 : i32]} : (!quir.qubit<1>) -> ()
  }
  // CHECK: } {quir.operatedQubits = [1 : i32, 2 : i32, 3 : i32]}
  return
}

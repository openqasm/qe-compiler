// RUN: qss-compiler -X=mlir --enable-circuits=true --extract-circuits %s | FileCheck %s
//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

module {
  oq3.declare_variable @obs : !quir.cbit<4>
  func @x(%arg0: !quir.qubit<1>) attributes {quir.classicalOnly = false} {
    return
  }
  // CHECK: quir.circuit @circuit_0
  // CHECK: quir.delay %arg0, (%arg1)
  // CHECK: %0:2 = quir.measure(%arg2, %arg3)
  // CHECK: quir.return %0#0, %0#1 : i1, i1
  // CHECK: quir.circuit @circuit_1
  // CHECK: quir.call_gate @x(%arg0)
  // CHECK: quir.return
  // CHECK: func @main()
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i1 = arith.constant 0 : i1
    %c0_i32 = arith.constant 0 : i32
    %c0_i4 = arith.constant 0 : i4
    %dur = quir.constant #quir.duration<1.000000e+00 : <ms>>
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %dur_0 = quir.constant #quir.duration<2.800000e+03 : <dt>>
    qcs.init
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      quir.delay %dur, () : !quir.duration<ms>, () -> ()
      qcs.shot_init {qcs.num_shots = 1000 : i32}
      %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      %3 = "oq3.cast"(%c0_i4) : (i4) -> !quir.cbit<4>
      oq3.variable_assign @obs : !quir.cbit<4> = %3
      quir.delay %dur_0, (%1) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
      %4:2 = quir.measure(%0, %2) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK-NOT: quir.delay %dur_0, (%1) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
      // CHECK-NOT: %4:2 = quir.measure(%0, %2) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK: %4:2 = quir.call_circuit @circuit_0(%dur_0, %1, %0, %2)
      qcs.parallel_control_flow {
      // CHECK: qcs.parallel_control_flow
      scf.if %4#0 {
      // scf.if
        quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
        // CHECK-NOT:  quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
        // CHECK: quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> ()
      } {quir.classicalOnly = false, quir.physicalIds = [0 : i32]}
      oq3.cbit_assign_bit @obs<4> [0] : i1 = %4#1
      } {quir.maxDelayCycles = 100 : i64, quir.physicalIds = [0 : i32]}
      quir.switch %c0_i32 {
      // CHECK: quir.switch
        } [
            1: {
                quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
                // CHECK-NOT:  quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
                // CHECK: quir.call_circuit @circuit_2(%1) : (!quir.qubit<1>) -> ()
            }
            2: {
                quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
                // CHECK-NOT:  quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
                // CHECK: quir.call_circuit @circuit_3(%2) : (!quir.qubit<1>) -> ()
            }
            3: {
            }
        ]
      scf.while : () -> () {
      // CHECK: scf.while
        quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
        // CHECK-NOT:  quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
        // CHECK: quir.call_circuit @circuit_4(%1) : (!quir.qubit<1>) -> ()
        scf.condition(%c0_i1)
        // CHECK: scf.condition
      } do {
        quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
        // CHECK-NOT:  quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
        // CHECK: quir.call_circuit @circuit_5(%2) : (!quir.qubit<1>) -> ()
        scf.yield
        // CHECK: scf.yield
      }
      %5 = quir.measure(%0) : (!quir.qubit<1>) -> (i1)
      // CHECK: %5 = quir.call_circuit @circuit_6(%0) : (!quir.qubit<1>) -> i1
      // CHECK-NOT: %5 = quir.measure(%0) : (!quir.qubit<1>) -> (i1)
      qcs.synchronize %0 : (!quir.qubit<1>) -> ()
      %6 = quir.measure(%0) : (!quir.qubit<1>) -> (i1)
      // CHECK: %6 = quir.call_circuit @circuit_7(%0) : (!quir.qubit<1>) -> i1
       // CHECK-NOT: %6 = quir.measure(%0) : (!quir.qubit<1>) -> (i1)
    } {qcs.shot_loop, quir.classicalOnly = false, quir.physicalIds = [0 : i32, 1 : i32, 2 : i32]}
    qcs.finalize
    return %c0_i32 : i32
  }
}

// RUN: qss-compiler -X=mlir --canonicalize --subroutine-cloning --quantum-decorate --merge-circuit-measures-topological --remove-unused-circuits %s | FileCheck %s

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

// This test is based on merge-measures.mlir but uses circuits

module {
  oq3.declare_variable @c0 : !quir.cbit<1>
  oq3.declare_variable @c1 : !quir.cbit<1>
  oq3.declare_variable @c2 : !quir.cbit<1>
  quir.circuit @circuit_cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1> ) {
    quir.call_gate @cx(%arg0, %arg1) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    quir.return
  }
  quir.circuit @circuit_0(%arg0: !quir.qubit<1> ) -> i1 {
    quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  quir.circuit @circuit_1(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  quir.circuit @circuit_2(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  quir.circuit @circuit_3(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  quir.circuit @circuit_4(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  quir.circuit @circuit_5(%arg0: !quir.qubit<1> ) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  quir.circuit @circuit_6(%arg0: !quir.qubit<1> ) {
    quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
    quir.return
  }
  // func.func @main() -> i32  {
  func.func @main() -> i32  {
    %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
    %q4 = quir.declare_qubit {id = 4 : i32} : !quir.qubit<1>
    %q5 = quir.declare_qubit {id = 5 : i32} : !quir.qubit<1>

    // one
    // CHECK:  %{{.*}} = quir.call_circuit @circuit_0_q0(%{{.}}) : (!quir.qubit<1>) -> i1
    %res = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> (i1)

    // CHECK:  quir.barrier %0, %1, %2, %3, %4, %5
    quir.barrier %q0, %q1, %q2, %q3, %q4, %q5: (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

    // two
    // CHECK:  %{{.*}}:2 = quir.call_circuit @"circuit_0_q0_circuit_1_q1+m"(%{{.}}, %{{.}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
    %res7_0 = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> (i1)
    %cast7_0 = "oq3.cast"(%res7_0) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast7_0
    %res7_1 = quir.call_circuit @circuit_1(%q1) : (!quir.qubit<1>) -> (i1)
    %cast7_1 = "oq3.cast"(%res7_0) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast7_1

    // CHECK:  quir.barrier %0, %1, %2, %3, %4, %5
    quir.barrier %q0, %q1, %q2, %q3, %q4, %q5: (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

    // three
    // CHECK:  %{{.*}}:3 = quir.call_circuit @"circuit_0_q0_circuit_1_q1_circuit_2_q2+m1+m"(%{{.}}, %{{.}}, %{{.}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1)
    %res6_0 = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> (i1)
    %cast6_0 = "oq3.cast"(%res6_0) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast6_0
    %res1 = quir.call_circuit @circuit_1(%q1) : (!quir.qubit<1>) -> (i1)
    %cast6_1 = "oq3.cast"(%res6_0) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast6_1
    %res2 = quir.call_circuit @circuit_2(%q2) : (!quir.qubit<1>) -> (i1)

    // CHECK:  quir.barrier %0, %1, %2, %3, %4, %5
    quir.barrier %q0, %q1, %q2, %q3, %q4, %q5: (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

    // four
    // CHECK:  %{{.*}}:4 = quir.call_circuit @"circuit_0_q0_circuit_1_q1_circuit_2_q2_circuit_3_q3+m0+m+m"(%{{.}}, %{{.}}, %{{.}}, %{{.}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
    %res5_0 = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> (i1)
    %cast5_0 = "oq3.cast"(%res5_0) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast5_0
    %res5_1 = quir.call_circuit @circuit_1(%q1) : (!quir.qubit<1>) -> (i1)
    %cast5_1 = "oq3.cast"(%res5_1) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast5_1
    %res5_2 = quir.call_circuit @circuit_2(%q2) : (!quir.qubit<1>) -> (i1)
    %cast5_2 = "oq3.cast"(%res5_2) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast5_2
    %res3 = quir.call_circuit @circuit_3(%q3) : (!quir.qubit<1>) -> (i1)

    // CHECK:  quir.barrier %0, %1, %2, %3, %4, %5
    quir.barrier %q0, %q1, %q2, %q3, %q4, %q5: (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

    // four_interrupted
    // CHECK:  %{{.*}}:4 = quir.call_circuit @"circuit_0_q0_circuit_1_q1_circuit_2_q2_circuit_3_q3+m+m+m"(%{{.}}, %{{.}}, %{{.}}, %{{.}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
    %res4_0 = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> (i1)
    %cast4_0 = "oq3.cast"(%res4_0) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast4_0
    %res4_1 = quir.call_circuit @circuit_1(%q1) : (!quir.qubit<1>) -> (i1)
    %cast4_1 = "oq3.cast"(%res4_1) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast4_1

    quir.barrier %q5 : (!quir.qubit<1>) -> ()

    // CHECK-NOT:  %{{.*}} = quir.call_circuit @circuit_2
    %res4_2 = quir.call_circuit @circuit_2(%q2) : (!quir.qubit<1>) -> (i1)
    %cast4_2 = "oq3.cast"(%res4_2) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %cast4_2
    %res4_3 = quir.call_circuit @circuit_3(%q3) : (!quir.qubit<1>) -> (i1)

    // CHECK:  quir.barrier %0, %1, %2, %3, %4, %5
    quir.barrier %q0, %q1, %q2, %q3, %q4, %q5: (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

    // inter_if
    // CHECK:  %{{.*}} = quir.call_circuit @circuit_0_q0(%0)
    %res3_0 =  quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> i1

    %cond = arith.constant 1 : i1

    scf.if %cond {
      quir.call_circuit @circuit_cx(%q2, %q3) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    }

    // CHECK:  %{{.*}}:2 = quir.call_circuit @"circuit_1_q1_circuit_2_q2+m0"(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
    %res3_1 = quir.call_circuit @circuit_1(%q1) : (!quir.qubit<1>) -> (i1)
    %res3_2 = quir.call_circuit @circuit_2(%q2) : (!quir.qubit<1>) -> (i1)

    // CHECK:  quir.barrier %0, %1, %2, %3, %4, %5
    quir.barrier %q0, %q1, %q2, %q3, %q4, %q5: (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

    // barrier
    // CHECK:  %{{.*}} = quir.call_circuit @circuit_0_q0(%{{.*}}) : (!quir.qubit<1>) -> i1
    %res2_0 = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> i1
    quir.barrier %q1, %q0 : (!quir.qubit<1>, !quir.qubit<1>) -> ()

    // CHECK:  %{{.*}}:2 = quir.call_circuit @"circuit_1_q1_circuit_2_q2+m"(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
    %res2_1 = quir.call_circuit @circuit_1(%q1) : (!quir.qubit<1>) -> i1

    quir.barrier %q0, %q3: (!quir.qubit<1>, !quir.qubit<1>) -> ()

    // CHECK-NOT:  %{{.*}} =  quir.call_circuit @circuit_2
    %res2_2 = quir.call_circuit @circuit_2(%q2) : (!quir.qubit<1>) -> i1

    // CHECK:  quir.barrier %0, %1, %2, %3, %4, %5
    quir.barrier %q0, %q1, %q2, %q3, %q4, %q5: (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

    // inter_switch
    // CHECK:  %{{.*}} = quir.call_circuit @circuit_0_q0(%0) : (!quir.qubit<1>) -> i1
    %res1_0 = quir.call_circuit @circuit_0(%q0) : (!quir.qubit<1>) -> i1

    %flag = arith.constant 0 : i32

    quir.switch %flag {
        // CHECK:  quir.call_circuit @circuit_6_q1(%{{.}}) : (!quir.qubit<1>) -> ()
        quir.call_circuit @circuit_6(%q1) : (!quir.qubit<1>) -> ()
    } [
        0: {
            // CHECK:  quir.call_circuit @circuit_6_q2(%{{.}}) : (!quir.qubit<1>) -> ()
            quir.call_circuit @circuit_6(%q2) : (!quir.qubit<1>) -> ()
        }
        1: {
            // CHECK:  quir.call_circuit @circuit_6_q3(%{{.}}) : (!quir.qubit<1>) -> ()
            quir.call_circuit @circuit_6(%q3) : (!quir.qubit<1>) -> ()
        }
    ]

    //  CHECK:  %{{.*}}:4 = quir.call_circuit @"circuit_2_q2_circuit_3_q3_circuit_4_q4_circuit_5_q5+m+m+m"(%2, %3, %4, %5) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
    %res1_1 = quir.call_circuit @circuit_2(%q2) : (!quir.qubit<1>) -> i1
    %res1_2 = quir.call_circuit @circuit_3(%q3) : (!quir.qubit<1>) -> i1

    quir.switch %flag {} [
        0: {
            quir.call_circuit @circuit_6(%q0) : (!quir.qubit<1>) -> ()
        }
    ]

    %res1_3 = quir.call_circuit @circuit_4(%q4) : (!quir.qubit<1>) -> (i1)
    %res1_4 = quir.call_circuit @circuit_5(%q5) : (!quir.qubit<1>) -> (i1)

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

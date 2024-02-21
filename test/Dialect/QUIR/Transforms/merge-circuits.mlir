// RUN: qss-compiler -X=mlir --subroutine-cloning --quantum-decorate --merge-circuits %s | FileCheck %s

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

module {
  oq3.declare_variable @qc0_meas : !quir.cbit<2>
  oq3.declare_variable {input} @p1 : f64
  oq3.declare_variable {input} @p2 : f64
  quir.circuit @circuit_0(%arg0: !quir.qubit<1>) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0: i1
  }
  quir.circuit @circuit_1(%arg1: !quir.qubit<1>) -> i1 {
    %0 = quir.measure(%arg1) : (!quir.qubit<1>) -> i1
    quir.return %0: i1
  }
  quir.circuit @circuit_2(%arg0: !quir.qubit<1>) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0: i1
  }
  quir.circuit @circuit_3(%arg1: !quir.qubit<1>) -> i1 {
    %0 = quir.measure(%arg1) : (!quir.qubit<1>) -> i1
    quir.return %0: i1
  }
  quir.circuit @circuit_4(%arg0: !quir.qubit<1>) -> i1 {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0: i1
  }
  quir.circuit @circuit_5(%arg0: !quir.qubit<1>,  %arg2: !quir.angle<64>) -> i1 {
    quir.call_gate @rz(%arg0, %arg2) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  quir.circuit @circuit_6(%arg0: !quir.qubit<1>) -> (i1, i1) {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    %1 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0, %1: i1, i1
  }
  quir.circuit @circuit_7(%arg0: !quir.qubit<1>) -> (i1, i1) {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    %1 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0, %1: i1, i1
  }
  quir.circuit @circuit_8(%arg0: !quir.qubit<1>) -> (i1, i1) {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    %1 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0, %1: i1, i1
  }
  quir.circuit @circuit_9(%arg0: !quir.qubit<1>) -> (i1, i1) {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    %1 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0, %1: i1, i1
  }
  // CHECK: @circuit_0_q0_circuit_1_q1(%arg0: !quir.qubit<1>
  // CHECK: %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
  // CHECK: %1 = quir.measure(%arg1) : (!quir.qubit<1>) -> i1
  // CHECK: quir.return %0, %1 : i1, i1
  // CHECK: }
  // CHECK: @circuit_8_q0_circuit_8_q0(%arg0: !quir.qubit<1>
  // CHECK: %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
  // CHECK: %1 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
  // CHECK: quir.barrier %arg0 : (!quir.qubit<1>) -> ()
  // CHECK: quir.barrier %arg1 : (!quir.qubit<1>) -> ()
  // CHECK: %2 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
  // CHECK: %3 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
  // CHECK: quir.return %0, %1, %2, %3 : i1, i1, i1, i1
  // CHECK: }
  func.func @main(%501 : i1) -> i32 {
    %true = arith.constant true
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %200 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    %201 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
    %202 = quir.declare_qubit {id = 4 : i32} : !quir.qubit<1>

    scf.if %501 {
      %2 = quir.call_circuit @circuit_0(%0) : (!quir.qubit<1>) -> i1
      %3 = quir.call_circuit @circuit_1(%1) : (!quir.qubit<1>) -> i1
      // CHECK-NOT: {{.*}} = quir.call_circuit @circuit_0(%0) : (!quir.qubit<1>) -> i1
      // CHECK-NOT: {{.*}} = quir.call_circuit @circuit_1(%1) : (!quir.qubit<1>) -> i1
      // CHECK: {{.*}}:2 = quir.call_circuit @circuit_0_q0_circuit_1_q1(%0, %1) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
    }

    scf.if %501 {
      // use oq3.cbit_assign_bit to prevent circuit merge with barrier
      oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %true
      %4 = quir.call_circuit @circuit_2(%0) : (!quir.qubit<1>) -> i1
      oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %4
      %5 = quir.call_circuit @circuit_3(%1) : (!quir.qubit<1>) -> i1
      oq3.cbit_assign_bit @qc0_meas<2> [1] : i1 = %5
      // CHECK-NOT: %4 = quir.call_circuit @circuit_2(%0) : (!quir.qubit<1>) -> i1
      // CHECK-NOT: oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %4
      // CHECK-NOT: %5 = quir.call_circuit @circuit_3(%1) : (!quir.qubit<1>) -> i1
      // CHECK-NOT: oq3.cbit_assign_bit @qc0_meas<2> [1] : i1 = %5
      // CHECK: %[[MEAS2:.*]]:2 = quir.call_circuit @circuit_2_q0_circuit_3_q1(%0, %1) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK-NOT: oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %[[MEAS2]]:0
      // CHECK-NOT: oq3.cbit_assign_bit @qc0_meas<2> [1] : i1 = %[[MEAS2]]:1
    }

    scf.if %501 {
      oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %true
      %6 = quir.call_circuit @circuit_4(%0) : (!quir.qubit<1>) -> i1
      %7 = oq3.variable_load @p1 : f64
      %8 = "oq3.cast"(%7) : (f64) -> !quir.angle<64>
      quir.barrier %200 : (!quir.qubit<1>) -> ()
      %9 = oq3.variable_load @p2 : f64
      %10 = "oq3.cast"(%9) : (f64) -> !quir.angle<64>
      quir.barrier %201 : (!quir.qubit<1>) -> ()
      %11 = quir.call_circuit @circuit_5(%1, %8) : (!quir.qubit<1>, !quir.angle<64>) -> i1
      // CHECK-NOT: %6 = quir.call_circuit @circuit_4(%0) : (!quir.qubit<1>) -> i1
      // CHECK-NOT: %7 = oq3.variable_load @p1 : f64
      // CHECK-NOT: %8 = "oq3.cast"(%7) : (f64) -> !quir.angle<64>
      // CHECK-NOT: %9 = oq3.variable_load @p2 : f64
      // CHECK-NOT: %10 = "oq3.cast"(%9) : (f64) -> !quir.angle<64>
      // CHECK: %[[LOAD:.*]] = oq3.variable_load @p1 : f64
      // CHECK: %[[CAST:.*]] = "oq3.cast"(%[[LOAD]]) : (f64) -> !quir.angle<64>
      // CHECK: {{.*}} = oq3.variable_load @p2 : f64
      // CHECK: %{{.*}}:2 = quir.call_circuit @circuit_4_q0_circuit_5_q1(%0, %1, %[[CAST]]) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<64>) -> (i1, i1)
    }

    scf.if %501 {
      oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %true
      %12:2 = quir.call_circuit @circuit_6(%0) : (!quir.qubit<1>) -> (i1, i1)
      quir.barrier %200 : (!quir.qubit<1>) -> ()
      %13:2 = quir.call_circuit @circuit_6(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK: %{{.*}}:4 = quir.call_circuit @circuit_6_q0_circuit_6_q0(%0) : (!quir.qubit<1>) -> (i1, i1, i1, i1)
    }

    scf.if %501 {
      oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %true
      %14:2 = quir.call_circuit @circuit_7(%0) : (!quir.qubit<1>) -> (i1, i1)
      quir.barrier %200, %201 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
      quir.barrier %200, %202 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
      %15:2 = quir.call_circuit @circuit_7(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK: %{{.*}}:4 = quir.call_circuit @circuit_7_q0_circuit_7_q0(%0) : (!quir.qubit<1>) -> (i1, i1, i1, i1)
    }

    scf.if %501 {
      oq3.cbit_assign_bit @qc0_meas<2> [0] : i1 = %true
      %16:2 = quir.call_circuit @circuit_8(%0) : (!quir.qubit<1>) -> (i1, i1)
      quir.barrier %0 : (!quir.qubit<1>) -> ()
      // CHECK-NOT: quir.barrier
      quir.barrier %200 : (!quir.qubit<1>) -> ()
      // CHECK-NOT: quir.barrier
      %17:2 = quir.call_circuit @circuit_8(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK: %{{.*}}:4 = quir.call_circuit @circuit_8_q0_circuit_8_q0(%0, %2) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
    }

    scf.if %501 {
      %18:2 = quir.call_circuit @circuit_9(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK: %{{.*}}:2 = quir.call_circuit @circuit_9_q0(%0) : (!quir.qubit<1>) -> (i1, i1)
      %19 = quir.measure(%0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
      // CHECK: %{{.*}} = quir.measure(%0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
      %20:2 = quir.call_circuit @circuit_9(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK: %{{.*}}:2 = quir.call_circuit @circuit_9_q0(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK-NOT: %{{.*}}:4 = quir.call_circuit @circuit_9_q0_circuit_9_q0(%0) : (!quir.qubit<1>) -> (i1, i1, i1, i1)
    }

    // verify that a qcs.parallel_control_flow will prevent circuits from being merged
    scf.if %501 {
      %21:2 = quir.call_circuit @circuit_9(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK: %{{.*}}:2 = quir.call_circuit @circuit_9_q0(%0) : (!quir.qubit<1>) -> (i1, i1)
      qcs.parallel_control_flow {
      // CHECK: qcs.parallel_control_flow
        scf.if %21#0 {
        %22 = quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> (i1)
        } else {
          qcs.delay_cycles() {time = 12 : i64} : () -> ()
        } {quir.classicalOnly = false, quir.physicalIds = [0 : i32]}
      } {quir.maxDelayCycles = 12 : i64, quir.physicalIds = [0 : i32]}
      // CHECK: } {quir.maxDelayCycles = 12 : i64, quir.physicalIds = [0 : i32]}
      %23:2 = quir.call_circuit @circuit_9(%0) : (!quir.qubit<1>) -> (i1, i1)
      // CHECK: %{{.*}}:2 = quir.call_circuit @circuit_9_q0(%0) : (!quir.qubit<1>) -> (i1, i1)
    }

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

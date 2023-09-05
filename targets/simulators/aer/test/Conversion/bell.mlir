// RUN: touch test.cfg
// RUN: qss-compiler -X=mlir --emit=mlir --target simulator --config test.cfg %s --num-shots=1 --simulator-output-cregs --quir-eliminate-variables --simulator-quir-to-aer | FileCheck %s

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
//

module {
  oq3.declare_variable @c0 : !quir.cbit<1>
  oq3.declare_variable @c1 : !quir.cbit<1>
  func @cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    // Begin with the instruction calling @aer_state
    // CHECK: {{.*}} = llvm.call @aer_state() : () -> !llvm.ptr<i8>

    // Angles are defined here by the pass
    // CHECK: {{.*}} = arith.constant 1.57079632679 : f64
    // CHECK: {{.*}} = arith.constant 0.000000e+00 : f64
    // CHECK: {{.*}} = arith.constant 3.1415926535900001 : f64

    qcs.init
    // CHECK: llvm.call @aer_state_configure{{.*$}}
    // CHECK: llvm.call @aer_state_configure{{.*$}}
    // CHECK: llvm.call @aer_state_configure{{.*$}}

    qcs.shot_init {qcs.num_shots = 1 : i32}
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
    // Call `aer_state_initialize` after all `aer_allocate_qubits` have been done.
    // CHECK: {{.*}} = llvm.call @aer_state_initialize{{.*$}}
    %false = arith.constant false
    %2 = "oq3.cast"(%false) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c0 : !quir.cbit<1> = %2
    %false_0 = arith.constant false
    %3 = "oq3.cast"(%false_0) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @c1 : !quir.cbit<1> = %3
    %angle = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    %angle_1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %angle_2 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    quir.builtin_U %0, %angle, %angle_1, %angle_2 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // CHECK: llvm.call @aer_apply_u3{{.*$}}
    quir.builtin_CX %0, %1 : !quir.qubit<1>, !quir.qubit<1>
    // CHECK: llvm.call @aer_apply_cx{{.*$}}
    %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
    oq3.cbit_assign_bit @c0<1> [0] : i1 = %4
    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    // CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
    oq3.cbit_assign_bit @c1<1> [0] : i1 = %5
    qcs.finalize
    // CHECK: llvm.call @aer_state_finalize{{.*$}}
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

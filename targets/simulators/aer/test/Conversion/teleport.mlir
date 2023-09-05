// RUN: touch test.cfg
// RUN: qss-compiler -X=mlir --emit=mlir --target simulator --config test.cfg %s --num-shots=1 --break-reset --simulator-output-cregs --quir-eliminate-variables --simulator-quir-to-aer | FileCheck %s

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
  oq3.declare_variable @c : !quir.cbit<3>
  func @cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    // CHECK: {{.*}} = llvm.call @aer_state() : () -> !llvm.ptr<i8>
    
    // Angles are defined here by the pass
    // CHECK: {{.*}} = arith.constant 3.1415926535900001 : f64
    // CHECK: {{.*}} = arith.constant 0.000000e+00 : f64
    // CHECK: {{.*}} = arith.constant 1.57079632679 : f64
    // CHECK: {{.*}} = arith.constant 1.000000e-01 : f64
    // CHECK: {{.*}} = arith.constant 2.000000e-01 : f64
    // CHECK: {{.*}} = arith.constant 3.000000e-01 : f64

    qcs.init
    // CHECK: llvm.call @aer_state_configure{{.*$}}
    // CHECK: llvm.call @aer_state_configure{{.*$}}
    // CHECK: llvm.call @aer_state_configure{{.*$}}

    qcs.shot_init {qcs.num_shots = 1 : i32}
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
    %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
    %c0_i3 = arith.constant 0 : i3
    %3 = "oq3.cast"(%c0_i3) : (i3) -> !quir.cbit<3>
    oq3.variable_assign @c : !quir.cbit<3> = %3
    %angle = quir.constant #quir.angle<3.000000e-01 : !quir.angle<64>>
    %angle_0 = quir.constant #quir.angle<2.000000e-01 : !quir.angle<64>>
    %angle_1 = quir.constant #quir.angle<1.000000e-01 : !quir.angle<64>>
    quir.builtin_U %0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // CHECK: llvm.call @aer_apply_u3{{.*$}}
    %angle_2 = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    %angle_3 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %angle_4 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    quir.builtin_U %1, %angle_2, %angle_3, %angle_4 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // CHECK: llvm.call @aer_apply_u3{{.*$}}
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    // CHECK: llvm.call @aer_apply_cx{{.*$}}
    quir.builtin_CX %0, %1 : !quir.qubit<1>, !quir.qubit<1>
    // CHECK: llvm.call @aer_apply_cx{{.*$}}
    %angle_5 = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    %angle_6 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %angle_7 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    quir.builtin_U %0, %angle_5, %angle_6, %angle_7 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // CHECK: llvm.call @aer_apply_u3{{.*$}}
    %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
    oq3.cbit_assign_bit @c<3> [0] : i1 = %4
    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    // CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
    oq3.cbit_assign_bit @c<3> [1] : i1 = %5
    %6 = oq3.variable_load @c : !quir.cbit<3>
    %7 = oq3.cbit_extractbit(%6 : !quir.cbit<3>) [0] : i1
    %c1_i32 = arith.constant 1 : i32
    %8 = "oq3.cast"(%7) : (i1) -> i32
    %9 = arith.cmpi eq, %8, %c1_i32 : i32
    scf.if %9 {
      %angle_9 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
      %angle_10 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
      %angle_11 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
      quir.builtin_U %2, %angle_9, %angle_10, %angle_11 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
      // CHECK: llvm.call @aer_apply_u3{{.*$}}
    }
    %10 = oq3.variable_load @c : !quir.cbit<3>
    %11 = oq3.cbit_extractbit(%10 : !quir.cbit<3>) [1] : i1
    %c1_i32_8 = arith.constant 1 : i32
    %12 = "oq3.cast"(%11) : (i1) -> i32
    %13 = arith.cmpi eq, %12, %c1_i32_8 : i32
    scf.if %13 {
      %angle_9 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
      %angle_10 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
      %angle_11 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
      quir.builtin_U %2, %angle_9, %angle_10, %angle_11 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
      // CHECK: llvm.call @aer_apply_u3{{.*$}}
    }
    qcs.finalize
    // CHECK: llvm.call @aer_state_finalize{{.*$}}
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

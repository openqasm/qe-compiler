// RUN: touch test.cfg
// Use default configuration values.
// RUN: echo "{}" >> test.cfg
// RUN: qss-compiler -X=mlir --emit=mlir --target aer-simulator --config test.cfg %s --simulator-output-cregs | FileCheck %s

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

  // CHECK: llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  
  func @main() -> i32 {
    qcs.init
    qcs.shot_init {qcs.num_shots = 1 : i32}
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
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
    quir.builtin_CX %0, %1 : !quir.qubit<1>, !quir.qubit<1>
    %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    oq3.cbit_assign_bit @c0<1> [0] : i1 = %4
    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    oq3.cbit_assign_bit @c1<1> [0] : i1 = %5

    // CHECK:      func @main() -> i32 {
    // CHECK-NEXT:   qcs.init
    // CHECK-NEXT:   qcs.shot_init {qcs.num_shots = 1 : i32}
    // CHECK-NEXT:   %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // CHECK-NEXT:   %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    // CHECK-NEXT:   %false = arith.constant false
    // CHECK-NEXT:   %2 = "oq3.cast"(%false) : (i1) -> !quir.cbit<1>
    // CHECK-NEXT:   oq3.variable_assign @c0 : !quir.cbit<1> = %2
    // CHECK-NEXT:   %false_0 = arith.constant false
    // CHECK-NEXT:   %3 = "oq3.cast"(%false_0) : (i1) -> !quir.cbit<1>
    // CHECK-NEXT:   oq3.variable_assign @c1 : !quir.cbit<1> = %3
    // CHECK-NEXT:   %angle = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    // CHECK-NEXT:   %angle_1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    // CHECK-NEXT:   %angle_2 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    // CHECK-NEXT:   quir.builtin_U %0, %angle, %angle_1, %angle_2 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // CHECK-NEXT:   quir.builtin_CX %0, %1 : !quir.qubit<1>, !quir.qubit<1>
    // CHECK-NEXT:   %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // CHECK-NEXT:   oq3.cbit_assign_bit @c0<1> [0] : i1 = %4
    // CHECK-NEXT:   %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    // CHECK-NEXT:   oq3.cbit_assign_bit @c1<1> [0] : i1 = %5

    // CHECK: %[[C0:.*]] = oq3.variable_load @c0 : !quir.cbit<1>
    // CHECK-NEXT: %[[RES0:.*]] = oq3.cbit_extractbit(%[[C0]] : !quir.cbit<1>) [0] : i1
    // CHECK: llvm.call @printf({{.*}}, %[[RES0]]) : (!llvm.ptr<i8>, i1) -> i32
    // CHECK: %[[C1:.*]] = oq3.variable_load @c1 : !quir.cbit<1>
    // CHECK-NEXT: %[[RES1:.*]] = oq3.cbit_extractbit(%[[C1]] : !quir.cbit<1>) [0] : i1
    // CHECK: llvm.call @printf({{.*}}, %[[RES1]]) : (!llvm.ptr<i8>, i1) -> i32

    qcs.finalize
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
    // CHECK: qcs.finalize
    // CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
    // CHECK-NEXT: return %c0_i32 : i32
  }
}

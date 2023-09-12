// RUN: touch test.cfg
// RUN: echo "{}" >> test.cfg
// RUN: qss-compiler -X=mlir --emit=mlir --target aer-simulator --config test.cfg %s --num-shots=1 --simulator-quir-to-aer | FileCheck %s

//
// This file was generated by:
// $ qss-compiler -X=mlir --emit=mlir --target aer-simulator --config test.cfg teleport.qasm \
// $   --simulator-output-cregs --quir-eliminate-variables
//

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
  llvm.mlir.global private constant @str_creg_c("  c : \00")
  llvm.mlir.global private constant @str_digit("%d\00")
  llvm.mlir.global private constant @str_endline("\0A\00")
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  func @cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    // CHECK: {{.*}} = llvm.call @aer_state() : () -> !llvm.ptr<i8>

    %c0_i3 = arith.constant 0 : i3
    %angle = quir.constant #quir.angle<3.000000e-01 : !quir.angle<64>>
    %angle_0 = quir.constant #quir.angle<2.000000e-01 : !quir.angle<64>>
    %angle_1 = quir.constant #quir.angle<1.000000e-01 : !quir.angle<64>>
    %angle_2 = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    %angle_3 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %angle_4 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    // CHECK: %[[angle:.*]] = arith.constant 3.000000e-01 : f64
    // CHECK-NEXT: %[[angle0:.*]] = arith.constant 2.000000e-01 : f64
    // CHECK-NEXT: %[[angle1:.*]] = arith.constant 1.000000e-01 : f64
    // CHECK-NEXT: %[[angle2:.*]] = arith.constant 1.57079632679 : f64
    // CHECK-NEXT: %[[angle3:.*]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT: %[[angle4:.*]] = arith.constant 3.1415926535900001 : f64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloca() : memref<i3>

    qcs.init
    // CHECK: llvm.call @aer_state_configure{{.*$}}
    // CHECK-NEXT: llvm.call @aer_state_configure{{.*$}}
    // CHECK-NEXT: llvm.call @aer_state_configure{{.*$}}

    qcs.shot_init {qcs.num_shots = 1 : i32}
    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %3 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    // CHECK: %[[q0:.*]] = llvm.call @aer_allocate_qubits{{.*$}}
    // CHECK: %[[q1:.*]] = llvm.call @aer_allocate_qubits{{.*$}}
    // CHECK: %[[q2:.*]] = llvm.call @aer_allocate_qubits{{.*$}}
    quir.builtin_U %1, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    quir.builtin_U %2, %angle_2, %angle_3, %angle_4 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    quir.builtin_CX %2, %3 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_U %1, %angle_2, %angle_3, %angle_4 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // CHECK: llvm.call @aer_apply_u3({{.*}}, %[[q0]], %[[angle]], %[[angle0]], %[[angle1]]) : (!llvm.ptr<i8>, i64, f64, f64, f64) -> ()
    // CHECK: llvm.call @aer_apply_u3({{.*}}, %[[q1]], %[[angle2]], %[[angle3]], %[[angle4]]) : (!llvm.ptr<i8>, i64, f64, f64, f64) -> ()
    // CHECK: llvm.call @aer_apply_cx({{.*}}, %[[q1]], %[[q2]]) : (!llvm.ptr<i8>, i64, i64) -> ()
    // CHECK: llvm.call @aer_apply_cx({{.*}}, %[[q0]], %[[q1]]) : (!llvm.ptr<i8>, i64, i64) -> ()
    // CHECK: llvm.call @aer_apply_u3({{.*}}, %[[q0]], %[[angle2]], %[[angle3]], %[[angle4]]) : (!llvm.ptr<i8>, i64, f64, f64, f64) -> ()
    %4 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    // CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
    %5 = oq3.cbit_insertbit(%c0_i3 : i3) [0] = %4 : i3
    %6 = quir.measure(%2) : (!quir.qubit<1>) -> i1
    // CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
    %7 = oq3.cbit_insertbit(%5 : i3) [1] = %6 : i3
    affine.store %7, %0[] : memref<i3>
    %8 = "oq3.cast"(%4) : (i1) -> i32
    %9 = arith.cmpi eq, %8, %c1_i32 : i32
    scf.if %9 {
      // CHECK: scf.if {{.*}} {
      quir.builtin_U %3, %angle_3, %angle_3, %angle_4 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
      // CHECK: llvm.call @aer_apply_u3({{.*}}, %[[q2]], %[[angle3]], %[[angle3]], %[[angle4]]) : (!llvm.ptr<i8>, i64, f64, f64, f64) -> ()
      // CHECK-NEXT: }
    }
    %10 = "oq3.cast"(%6) : (i1) -> i32
    %11 = arith.cmpi eq, %10, %c1_i32 : i32
    scf.if %11 {
      // CHECK: scf.if {{.*}} {
      quir.builtin_U %3, %angle_4, %angle_3, %angle_4 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
      // CHECK: llvm.call @aer_apply_u3({{.*}}, %[[q2]], %[[angle4]], %[[angle3]], %[[angle4]]) : (!llvm.ptr<i8>, i64, f64, f64, f64) -> ()
      // CHECK-NEXT: }
    }
    %12 = llvm.mlir.addressof @str_endline : !llvm.ptr<array<2 x i8>>
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.getelementptr %12[%13, %13] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %15 = llvm.mlir.addressof @str_digit : !llvm.ptr<array<3 x i8>>
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.getelementptr %15[%16, %16] : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %18 = llvm.mlir.addressof @str_creg_c : !llvm.ptr<array<7 x i8>>
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.getelementptr %18[%19, %19] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %21 = llvm.call @printf(%20) : (!llvm.ptr<i8>) -> i32
    %22 = affine.load %0[] : memref<i3>
    %23 = oq3.cbit_extractbit(%22 : i3) [2] : i1
    %24 = llvm.call @printf(%17, %23) : (!llvm.ptr<i8>, i1) -> i32
    %25 = oq3.cbit_extractbit(%22 : i3) [1] : i1
    %26 = llvm.call @printf(%17, %25) : (!llvm.ptr<i8>, i1) -> i32
    %27 = oq3.cbit_extractbit(%22 : i3) [0] : i1
    %28 = llvm.call @printf(%17, %27) : (!llvm.ptr<i8>, i1) -> i32
    %29 = llvm.call @printf(%14) : (!llvm.ptr<i8>) -> i32
    qcs.finalize
    return %c0_i32 : i32
  }
}


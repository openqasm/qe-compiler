// RUN: qss-compiler -X=mlir --canonicalize --quir-eliminate-variables %s --canonicalize | FileCheck %s
//
// This test verifies that there is no store-forwarding where control-flow makes
// it impossible.

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

// CHECK: module
module {
  oq3.declare_variable @b : !quir.cbit<1>
  func.func @x(%arg0: !quir.qubit<1>) {
    return
  }
  func.func @main() -> i32 {
    // CHECK-DAG: [[MEMREF:%.*]] = memref.alloca() : memref<i1>
    // CHECK-DAG: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32}
    // CHECK-DAG: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32}
    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

    %false = arith.constant false
    %4 = "oq3.cast"(%false) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @b : !quir.cbit<1> = %4

    // CHECK: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]])
    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1

    // CHECK: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]])
    // CHECK: affine.store [[MEASURE1]], [[MEMREF]]
    %6 = quir.measure(%2) : (!quir.qubit<1>) -> i1
    oq3.cbit_assign_bit @b<1> [0] : i1 = %6

    // A variable update inside a control flow branch currently cannot be
    // simplified. Thus the store and load operations must be kept.
    scf.if %5 {
      quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
      oq3.cbit_assign_bit @b<1> [0] : i1 = %5
    }

    // CHECK: [[LOAD:%.*]] = affine.load [[MEMREF]]
    %10 = oq3.variable_load @b : !quir.cbit<1>
    %c1_i32_1 = arith.constant 1 : i32
    %11 = "oq3.cast"(%10) : (!quir.cbit<1>) -> i32

    %12 = arith.cmpi eq, %11, %c1_i32_1 : i32
    // CHECK: scf.if [[LOAD]]
    scf.if %12 {
      quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
    }

    return %11 : i32
  }
}

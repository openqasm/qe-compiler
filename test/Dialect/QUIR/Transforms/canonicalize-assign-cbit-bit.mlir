// RUN: qss-compiler -X=mlir --canonicalize %s | FileCheck %s --implicit-check-not cbit_assign_bit
//
// This test case validates that the single-bit cbit assignments
// are simplified to variable assignments by
// AssignSingleCbitToAssignVariablePattern.

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
  oq3.declare_variable @a : !quir.cbit<1>
  oq3.declare_variable @b : !quir.cbit<1>

  func @main() -> i32 {
    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

    // CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0
    // CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1

    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    oq3.cbit_assign_bit @a<1> [0] : i1 = %5

    // CHECK: [[MEAS0:%.*]] = quir.measure([[QUBIT0]])
    // CHECK: [[CAST0:%.*]] = "oq3.cast"([[MEAS0]]) : (i1) -> !quir.cbit<1>
    // CHECK: oq3.variable_assign @a : !quir.cbit<1> = [[CAST0]]

    %6 = quir.measure(%2) : (!quir.qubit<1>) -> i1
    oq3.cbit_assign_bit @b<1> [0] : i1 = %6

    // CHECK: [[MEAS1:%.*]] = quir.measure([[QUBIT1]])
    // CHECK: [[CAST1:%.*]] = "oq3.cast"([[MEAS1]]) : (i1) -> !quir.cbit<1>
    // CHECK: oq3.variable_assign @b : !quir.cbit<1> = [[CAST1]]

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

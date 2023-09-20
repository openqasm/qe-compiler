// RUN: qss-compiler --reorder-measures %s | FileCheck %s

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

// This regression test case validates that reorder-measures can move
// variable_load operations out of the way of a reorder-measurement

// based on failures for
// %26 = quir.measure(%0) : (!quir.qubit<1>) -> i1
// %27 = oq3.variable_load @p017 : !quir.angle<64>
// quir.call_gate @rz(%4, %27) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// %28 = quir.measure(%4) : (!quir.qubit<1>) -> i1
//
// This caused issues with measurement reordering being unable to merge
// some measurements when input keywords were used

module {
  oq3.declare_variable {input} @p001 : !quir.angle<64>
  oq3.declare_variable {input} @p002 : !quir.angle<64>
  oq3.declare_variable {input} @p003 : !quir.angle<64>
  func @rz(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) attributes {quir.classicalOnly = false} {
    return
  }
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i32 = arith.constant 0 : i32
    %angle = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %0 = quir.declare_qubit {id = 12 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 16 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 22 : i32} : !quir.qubit<1>
    %3 = quir.declare_qubit {id = 23 : i32} : !quir.qubit<1>
    
    oq3.variable_assign @p001 : !quir.angle<64> = %angle

    // test do not reorder if variable_assign is between measures
    %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    oq3.variable_assign @p002 : !quir.angle<64> = %angle
    %5 = oq3.variable_load @p002 : !quir.angle<64>
    quir.call_gate @rz(%1, %5) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    
    // CHECK: %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // CHECK: oq3.variable_assign @p002 : !quir.angle<64> = %angle
    // CHECK: %5 = oq3.variable_load @p002 : !quir.angle<64>
    // CHECK: quir.call_gate @rz(%1, %5) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    
    // test re-order if:
    // variable_assign not in block || 
    // variable_assign above measure in block

    %6 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    // CHECK-NOT: %6 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    
    %7 = oq3.variable_load @p003 : !quir.angle<64>
    quir.call_gate @rz(%2, %7) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    %8 = oq3.variable_load @p001 : !quir.angle<64>
    quir.call_gate @rz(%3, %8) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    
    // CHECK: %6 = oq3.variable_load @p003 : !quir.angle<64>
    // CHECK: quir.call_gate @rz(%2, %6) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    // CHECK: %7 = oq3.variable_load @p001 : !quir.angle<64>
    // CHECK: quir.call_gate @rz(%3, %7) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    // CHECK: %8 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    %9 = quir.measure(%3) : (!quir.qubit<1>) -> i1
    // CHECK: %9 = quir.measure(%3) : (!quir.qubit<1>) -> i1
    
    return %c0_i32 : i32
  }
}

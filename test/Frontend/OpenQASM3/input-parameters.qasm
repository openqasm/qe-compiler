OPENQASM 3;
// RUN: qss-compiler -X=qasm --emit=mlir --enable-parameters --enable-circuits %s | FileCheck %s --check-prefixes=CHECK,CHECK-XX

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

// This test case validates that input and output modifiers for variables are
// parsed correctly and are reflected in generated QUIR.

// TODO: putting resets in circuits has been disabled. The XX-tests 
// are the correct tests if it is re-enabled. The CHECK-XX should be removed
// in that case

qubit $0;
qubit $2;

gate sx q { }
gate rz(phi) q { }

input angle theta = 3.141;
// CHECK: qcs.declare_parameter @_QX64_5thetaEE_ : !quir.angle<64> = #quir.angle<3.141000e+00 : !quir.angle<64>>

input float[64] theta2 = 1.56;
// CHECK: qcs.declare_parameter @_QDDd64_6theta2EE_ : f64 = 1.560000e+00 : f64

reset $0;

sx $0;
rz(theta) $0;
sx $0;

bit b;

b = measure $0;

sx $0;
rz(theta2) $0;
sx $0;

bit c;

c = measure $0;

// CHECK: quir.circuit @circuit_0(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) {
// XX-CHECK-NEXT: quir.reset %arg0 : !quir.qubit<1>
// CHECK-NEXT: quir.call_gate @sx(%arg0) : (!quir.qubit<1>) -> ()
// CHECK: quir.return 
// CHECK-NEXT: } 

// CHECK: quir.circuit @circuit_1(%arg0: !quir.qubit<1>) -> i1 {
// CHECK-NEXT: %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
// CHECK-NEXT: quir.return %0 : i1
// CHECK-NEXT: }

// CHECK: func @main() -> i32 {
// CHECK: scf.for %arg0 = %c0 to %c1000 step %c1 {
// CHECK: %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: %1 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>

// CHECK: %2 = qcs.parameter_load @_QX64_5thetaEE_ : !quir.angle<64>
// CHECK: oq3.variable_assign @theta : !quir.angle<64> = %2
// CHECK: %3 = qcs.parameter_load @_QDDd64_6theta2EE_ : f64
// CHECK: oq3.variable_assign @theta2 : f64 = %3
// CHECK-XX: quir.reset %0 : !quir.qubit<1>
// CHECK-NOT: oq3.variable_assign @theta : !quir.angle<64> = %angle

// CHECK: quir.call_circuit @circuit_0(%0, %4) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK: %6 = quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> i1
// CHECK: oq3.cbit_assign_bit @b<1> [0] : i1 = %6

// CHECK: %7 = oq3.variable_load @theta2 : f64
// CHECK: %8 = "oq3.cast"(%7) : (f64) -> !quir.angle<64>
// CHECK: quir.call_circuit @circuit_2(%0, %8) : (!quir.qubit<1>, !quir.angle<64>) -> ()

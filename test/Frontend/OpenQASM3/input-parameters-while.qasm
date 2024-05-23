OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --enable-parameters --enable-circuits-from-qasm %s | FileCheck %s

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

gate rz(phi) q { }

input angle theta = 3.141;

qubit $0;
int n = 1;

bit is_excited;

// CHECK: func.func @h(%arg0: !quir.qubit<1>) {
// CHECK: quir.call_circuit @circuit_0(%arg0) : (!quir.qubit<1>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK: quir.circuit @circuit_0(%arg0: !quir.qubit<1>) {
// CHECK-NEXT: %angle = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
// CHECK-NEXT: %angle_0 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// CHECK-NEXT: %angle_1 = quir.constant #quir.angle<3.1415926535900001> : !quir.angle<64>
// CHECK-NEXT: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// CHECK-NEXT: quir.return
// CHECK-NEXT: }

// CHECK: func.func @rz(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) {
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK: quir.circuit @circuit_1(%arg0: !quir.qubit<1>) -> i1 {
// CHECK-NEXT: quir.call_gate @h(%arg0) : (!quir.qubit<1>) -> ()
// CHECK %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
// CHECK: quir.return %0 : i1
// CHECK-NEXT: }

// CHECK: quir.circuit @circuit_2(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) {
// CHECK-NEXT: quir.call_gate @h(%arg0) : (!quir.qubit<1>) -> ()
// CHECK quir.call_gate @rz(%arg0, %arg1) : (!quir.qubit<1>) -> ()
// CHECK: quir.return
// CHECK-NEXT: }

// CHECK: func.func @main() -> i32 {
// CHECK: scf.for %arg0 = %c0 to %c1000 step %c1 {
// CHECK: {{.*}} = qcs.parameter_load "input_theta" : !quir.angle<64> {initial_value = 0.000000e+00 : f64}
// CHECK: [[QUBIT:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: scf.while : () -> () {
// CHECK:    [[N:%.*]] = oq3.variable_load @n : i32
// CHECK:     %c0_i32_0 = arith.constant 0 : i32
// CHECK:     [[COND:%.*]] = arith.cmpi ne, [[N]], %c0_i32_0 : i32
// CHECK:     scf.condition([[COND]])
// CHECK: } do {
while (n != 0) {
    h $0;
    is_excited = measure $0;
    // CHECK: [[MEASURE:%.*]] = quir.call_circuit @circuit_1([[QUBIT]]) : (!quir.qubit<1>) -> i1
    // CHECK: oq3.cbit_assign_bit @is_excited<1> [0] : i1 = [[MEASURE]]
    // CHECK: [[EXCITED:%.*]] = oq3.variable_load @is_excited : !quir.cbit<1>
    // CHECK: [[COND2:%.*]] = "oq3.cast"([[EXCITED]]) : (!quir.cbit<1>) -> i1

    // CHECK: scf.if [[COND2]] {
    if (is_excited) {
        // CHECK:  [[THETA:%.*]] = oq3.variable_load @theta : !quir.angle<64>
        // CHECK:  quir.call_circuit @circuit_2([[QUBIT]], [[THETA]]) : (!quir.qubit<1>, !quir.angle<64>) -> ()
        // CHECK: }
        h $0;
        rz(theta) $0;
    }
    // error: Binary operation ASTOpTypeSub not supported yet.
    // n = n - 1;
    // CHECK: %c0_i32_0 = arith.constant 0 : i32
    // CHECK: oq3.variable_assign @n : i32 = %c0_i32_0
    n = 0;  // workaround for n = n - 1
    // CHECK: scf.yield
}

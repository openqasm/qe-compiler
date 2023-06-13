OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir --enable-parameters --enable-circuits %s | FileCheck %s

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


qubit $2;
qubit $3;

bit is_excited;
bit other;
bit result;

gate x q { } 
gate rz(phi) q { }

input angle theta = 3.141;
// CHECK: qcs.declare_parameter @_QX64_5thetaEE_ : !quir.angle<64> = #quir.angle<3.141000e+00 : !quir.angle<64>>

x $2;
rz(theta) $2;
x $3;

is_excited = measure $2;


// CHECK: quir.circuit @circuit_0(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>, %arg2: !quir.angle<64>) -> i1 {
// CHECK-NEXT: quir.call_gate @x(%arg1) : (!quir.qubit<1>) -> ()
// CHECK: %0 = quir.measure(%arg1) : (!quir.qubit<1>) -> i1
// CHECK-NEXT: quir.return %0 : i1
// CHECK: }

// CHECK: quir.circuit @circuit_1(%arg0: !quir.qubit<1>) -> i1 {
// CHECK-NEXT: %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
// CHECK-NEXT: quir.return %0 : i1
// CHECK-NEXT: }

// CHECK: func @main() -> i32 {
// CHECK: scf.for %arg0 = %c0 to %c1000 step %c1 {
// CHECK: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// CHECK: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>

// CHECK: [[EXCITED:%.*]] = oq3.variable_load @is_excited : !quir.cbit<1>
// CHECK: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// CHECK: [[EXCITEDCAST:%[0-9]+]] = "oq3.cast"([[EXCITED]]) : (!quir.cbit<1>) -> i32
// CHECK: [[COND0:%.*]] = arith.cmpi eq, [[EXCITEDCAST]], [[CONST]] : i32
// CHECK: scf.if [[COND0]] {
if (is_excited == 1) {
// CHECK: [[MEASURE3:%.*]] = quir.call_circuit @circuit_1([[QUBIT3]]) : (!quir.qubit<1>) -> i1
// CHECK: oq3.cbit_assign_bit @other<1> [0] : i1 = [[MEASURE3]]
  other = measure $3;
// CHECK: [[OTHER:%.*]] = oq3.variable_load @other : !quir.cbit<1>
// CHECK: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// CHECK: [[OTHERCAST:%[0-9]+]] = "oq3.cast"([[OTHER]]) : (!quir.cbit<1>) -> i32
// CHECK: [[COND1:%.*]] = arith.cmpi eq, [[OTHERCAST]], [[CONST]] : i32
// CHECK: scf.if [[COND1]] {
  if (other == 1){
// CHECK: [[THETA:%.*]] = oq3.variable_load @theta : !quir.angle<64>
// CHECK: quir.call_circuit @circuit_2([[QUBIT2]], [[THETA]]) : (!quir.qubit<1>, !quir.angle<64>) -> ()
     x $2;
     rz(theta) $2;
  }
}
// CHECK: [[MEASURE2:%.*]] = quir.call_circuit @circuit_3([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// CHECK: oq3.cbit_assign_bit @result<1> [0] : i1 = [[MEASURE2]]
result = measure $2;

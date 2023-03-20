OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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



// MLIR-DAG: oq3.declare_variable @a : i1
// MLIR-DAG: oq3.declare_variable @j : !quir.cbit<1>

// Angle
// MLIR: %{{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<3>>
angle[3] c = 0;
// MLIR: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;
// MLIR: %c1_i32{{.*}} = arith.constant 1 : i32
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=c, bits=3), right=IntNode(signed=true, value=1, bits=32))
c = 1;
// MLIR: %[[CAST_RESULT:.*]] = "oq3.cast"(%{{.*}}) : (i32) -> !quir.angle<3>
// MLIR:       oq3.variable_assign @c : !quir.angle<3> = %[[CAST_RESULT]]
// MLIR:       %[[USE_RESULT:.*]] = oq3.variable_load @c : !quir.angle<3>
// MLIR: %[[CAST_RESULT1:.*]] = "oq3.cast"(%[[USE_RESULT]]) : (!quir.angle<3>) -> i1
// MLIR: scf.if %[[CAST_RESULT1]] {
if (c) {
    U(0,0,0) $0;
}

// Boolean
// MLIR: %[[TRUE:.*]] = arith.constant true
// MLIR: oq3.variable_assign @a : i1 = %[[TRUE]]
bool a = true;
// MLIR: %[[FALSE:.*]] = arith.constant false
// MLIR: oq3.variable_assign @a : i1 = %[[FALSE]]
// ASR-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=a, bits=8), right=BoolNode(name=bool, false))
a = false;
// MLIR: [[A:%.*]] = oq3.variable_load @a : i1
// MLIR: scf.if [[A]] {
if (a) {
    U(0, 0, 0) $0;
}

// Bit
bit j = 1;
// MLIR: %{{.*}} = arith.constant 0 : i32
// ASR-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=j, bits=1), right=IntNode(signed=true, value=0, bits=32))
j = 0;
// MLIR: scf.if %{{.*}} {
if (j) {
    U(0, 0, 0) $0;
}

// Int
// MLIR: %{{.*}} = arith.constant 5 : i32
int[32] x = 5;
// MLIR: %{{.*}} = arith.constant 6 : i32
// ASR-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=x, bits=32), right=IntNode(signed=true, value=6, bits=32))
x = 6;
// MLIR: %[[CAST_RESULT4:.*]] = "oq3.cast"(%{{.*}}) : (i32) -> i1
// MLIR: scf.if %[[CAST_RESULT4]] {
if (x) {
    U(0, 0, 0) $0;
}

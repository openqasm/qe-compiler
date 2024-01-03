OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false | FileCheck %s --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s --check-prefixes MLIR,MLIR-CIRCUITS

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

// MLIR: module
// MLIR-DAG: func.func @main

// AST-PRETTY DeclarationNode(type=ASTTypeBitset, CBitNode(name=a, bits=1))
// DeclarationNode(type=ASTTypeBitset, CBitNode(name=b, bits=2, value=10))
// MLIR-DAG: oq3.declare_variable @a : !quir.cbit<1>
// MLIR-DAG: oq3.declare_variable @b : !quir.cbit<2>
bit a;
bit[2] b = "10";
int c = 5;

// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[CONSTCAST:%[0-9]+]] = "oq3.cast"([[CONST]]) : (i32) -> !quir.cbit<1>
// MLIR: oq3.variable_assign @a : !quir.cbit<1> = [[CONSTCAST]]
a = 1;

// MLIR: [[A:%.*]] = oq3.variable_load @a : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[ACAST:%[0-9]+]] = "oq3.cast"([[A]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND:%.*]] = arith.cmpi eq, [[ACAST]], [[CONST]] : i32
// MLIR: scf.if [[COND]] {
if (a == 1) {
// AST-PRETTY: IdentifierRefNode(name=b[0], IdentifierNode(name=b, bits=2), index=0)
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[CONSTCAST:%[0-9]+]] = "oq3.cast"([[CONST]]) : (i32) -> i1
// MLIR: oq3.cbit_assign_bit @b<2> [0] : i1 = [[CONSTCAST]]
  b[0] = 1;
}
// MLIR: }

// AST-PRETTY: IdentifierRefNode(name=b[1], IdentifierNode(name=b, bits=2), index=1)
// MLIR: [[B:%.*]] = oq3.variable_load @b : !quir.cbit<2>
// MLIR: [[BIT:%.*]] = oq3.cbit_extractbit([[B]] : !quir.cbit<2>) [1] : i1
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 0 : i32
// MLIR: [[BITCAST:%[0-9]+]] = "oq3.cast"([[BIT]]) : (i1) -> i32
// MLIR: [[COND:%.*]] = arith.cmpi eq, [[BITCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND]] {
if (b[1] == 0) {
}
// MLIR: }

bit[2] d;

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32}
qubit $0;

// MLIR-NO-CIRCUITS: [[MEASUREMENT:%.*]] = quir.measure([[QUBIT0]])
// MLIR-CIRCUITS: [[MEASUREMENT:%.*]] = quir.call_circuit @circuit_0([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @d<2> [0] : i1 = [[MEASUREMENT]]
d[0] = measure $0;

if (d[0] == 1) {
  d[1] = d[0];
}

d[1] = measure $0;

// AST-PRETTY: operand=OperatorNode(op=!, target-identifier=IdentifierRefNode(name=b[0], IdentifierNode(name=b, bits=2), index=0)
// MLIR: [[B:%.*]] = oq3.variable_load @b
// MLIR: [[B0:%.*]] = oq3.cbit_extractbit([[B]] : {{.*}}) [0]
// MLIR: [[NOTB0:%.*]] = {{.*}} [[B0]]
// MLIR: oq3.cbit_assign_bit @d<2> [0] : i1 = [[NOTB0]]
d[0] = ! b[0];

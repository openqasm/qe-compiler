OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
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

qubit $0;
// MLIR: %false = arith.constant false
bool a = false;
// MLIR: %true = arith.constant true
bool b = true;
// MLIR: %true_0 = arith.constant true
// MLIR: oq3.variable_assign @c : i1 = %true_0
bool c = 13;

// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeLogicalOr, left=IdentifierNode(name=a, bits=8), right=IdentifierNode(name=b, bits=8))
// AST-PRETTY: statements=
// AST-PRETTY: UGateOpNode{{.*}}
// AST-PRETTY: )
// MLIR: [[A:%.*]] = oq3.variable_load @a : i1
// MLIR: [[B:%.*]] = oq3.variable_load @b : i1
// MLIR: [[COND:%.*]] = arith.ori [[A]], [[B]] : i1
// MLIR: scf.if [[COND]] {
if (a || b) {
    U(0, 0, 0) $0;
}

// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeLogicalAnd, left=IdentifierNode(name=a, bits=8), right=IdentifierNode(name=b, bits=8))
// AST-PRETTY: statements=
// AST-PRETTY: UGateOpNode{{.*}}
// AST-PRETTY: )
// MLIR: [[A:%.*]] = oq3.variable_load @a : i1
// MLIR: [[B:%.*]] = oq3.variable_load @b : i1
// MLIR: [[COND:%.*]] = arith.andi [[A]], [[B]] : i1
// MLIR: scf.if [[COND]] {
if (a && b) {
    U(0, 0, 0) $0;
}

// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeLogicalAnd, left=BinaryOpNode(type=ASTOpTypeLogicalAnd, left=IdentifierNode(name=a, bits=8), right=IdentifierNode(name=b, bits=8))
// AST-PRETTY: , right=IdentifierNode(name=c, bits=8))
// AST-PRETTY: statements=
// AST-PRETTY: UGateOpNode{{.*}}
// AST-PRETTY: )
// MLIR-DAG: [[A:%.*]] = oq3.variable_load @a : i1
// MLIR-DAG: [[B:%.*]] = oq3.variable_load @b : i1
// MLIR-DAG: [[C:%.*]] = oq3.variable_load @c : i1
// MLIR-DAG: [[TMP:%.*]] = arith.andi [[A]], [[B]] : i1
// MLIR-DAG: [[COND:%.*]] = arith.andi [[TMP]], [[C]] : i1
// MLIR: scf.if [[COND]] {
if (a && b && c) {
    U(0, 0, 0) $0;
}

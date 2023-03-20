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
bit a;
measure $0 -> a;

// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=UnaryOpNode(type=ASTOpTypeLogicalNot, operand=OperatorNode(op=!, target-identifier=IdentifierNode(name=a, bits=1))
// MLIR: %[[A:.*]] = quir.use_variable @a : !quir.cbit<1>
// MLIR: %[[CAST1:.*]] = "quir.cast"(%[[A]]) : (!quir.cbit<1>) -> i1
// MLIR: %[[CMP1:.*]] = arith.cmpi ne, %[[CAST1]], %true{{.*}}  : i1
// MLIR: scf.if %[[CMP1]] {
if (!a) {
    U(0, 0, 0) $0;
}

bool b = true;
// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=UnaryOpNode(type=ASTOpTypeLogicalNot, operand=OperatorNode(op=!, target-identifier=IdentifierNode(name=b, bits=8))
// MLIR: %[[B:.*]] = quir.use_variable @b : i1
// MLIR-DAG: %[[CMP2:.*]] = arith.cmpi ne, %[[B]], %true{{.*}} : i1
// MLIR: scf.if %[[CMP2]] {
if (!b) {
    U(0, 0, 0) $0;
}

bit c;
c = 0;
// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=UnaryOpNode(type=ASTOpTypeLogicalNot, operand=OperatorNode(op=!, target-identifier=IdentifierNode(name=c, bits=1))
// MLIR: %[[C:.*]] = quir.use_variable @c : !quir.cbit<1>
// MLIR: %[[CAST4:.*]] = "quir.cast"(%[[C]]) : (!quir.cbit<1>) -> i1
// MLIR: %[[CMP3:.*]] = arith.cmpi ne, %[[CAST4]], %true{{.*}} : i1
// MLIR: scf.if %[[CMP3]] {
if (!c) {
    U(0, 0, 0) $0;
}

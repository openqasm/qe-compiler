OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm=false| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

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

// MLIR-DAG: oq3.declare_variable @condition_true : i1
// MLIR-DAG: oq3.declare_variable @condition_false : i1

// MLIR-CIRCUITS: quir.circuit @circuit_0(%[[ARG0:.*]]: !quir.qubit<1>) {
// MLIR-CIRCUITS: quir.builtin_U %[[ARG0]], %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.return

// MLIR-DAG: %true = arith.constant true
// MLIR-DAG: %false = arith.constant false
// MLIR-DAG: oq3.variable_assign @condition_true : i1 = %true
// MLIR-DAG: oq3.variable_assign @condition_false : i1 = %false
// AST-PRETTY: DeclarationNode(type=ASTTypeBool, BoolNode(name=condition_true, true))
// AST-PRETTY: DeclarationNode(type=ASTTypeBool, BoolNode(name=condition_false, false))
bool condition_true = true;
bool condition_false = false;

// MLIR: %false_0 = arith.constant false
bool my_bool;

// MLIR: [[CONDITION_TRUE:%.*]] = oq3.variable_load @condition_true : i1
// MLIR: scf.if [[CONDITION_TRUE]] {
// MLIR-NO-CIRCUITS: quir.builtin_U %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>
// MLIR-CIRCUITS: quir.call_circuit @circuit_0(%{{.*}}) : (!quir.qubit<1>) -> ()
// AST-PRETTY: condition=IdentifierNode(name=condition_true, bits=8),
if (condition_true) {
  U(1.57079632679, 0.0, 3.14159265359) $0;
}

// MLIR-DAG: [[COND_TRUE:%.*]] = oq3.variable_load @condition_true : i1
// MLIR-DAG: [[COND_FALSE:%.*]] = oq3.variable_load @condition_false : i1
// MLIR: [[OR:%.*]] = arith.ori [[COND_TRUE]], [[COND_FALSE]] : i1
// MLIR: oq3.variable_assign @my_bool : i1 = [[OR]]
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=my_bool, bits=8), right=BinaryOpNode(type=ASTOpTypeLogicalOr, left=IdentifierNode(name=condition_true, bits=8), right=IdentifierNode(name=condition_false, bits=8))
my_bool = condition_true || condition_false;

// MLIR: [[COND_TRUE:%.*]] = oq3.variable_load @condition_true : i1
// MLIR: [[CONST_TRUE:%.*]] = arith.constant true
// MLIR: [[NOT:%.*]] = arith.cmpi ne, [[COND_TRUE]], [[CONST_TRUE]] : i1
// MLIR: oq3.variable_assign @my_bool : i1 = [[NOT]]
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=my_bool, bits=8), right=UnaryOpNode(type=ASTOpTypeLogicalNot, operand=OperatorNode(op=!, target-identifier=IdentifierNode(name=condition_true, bits=8))
my_bool = !condition_true;

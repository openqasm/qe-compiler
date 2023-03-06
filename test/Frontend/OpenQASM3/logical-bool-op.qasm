OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// (C) Copyright IBM 2023.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

qubit $0;
// MLIR: %false = arith.constant false
bool a = false;
// MLIR: %true = arith.constant true
bool b = true;
// MLIR: %true_0 = arith.constant true
// MLIR: quir.assign_variable @c : i1 = %true_0
bool c = 13;

// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeLogicalOr, left=IdentifierNode(name=a, bits=8), right=IdentifierNode(name=b, bits=8))
// AST-PRETTY: statements=
// AST-PRETTY: UGateOpNode{{.*}}
// AST-PRETTY: )
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
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
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
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
// MLIR-DAG: [[A:%.*]] = quir.use_variable @a : i1
// MLIR-DAG: [[B:%.*]] = quir.use_variable @b : i1
// MLIR-DAG: [[C:%.*]] = quir.use_variable @c : i1
// MLIR-DAG: [[TMP:%.*]] = arith.andi [[A]], [[B]] : i1
// MLIR-DAG: [[COND:%.*]] = arith.andi [[TMP]], [[C]] : i1
// MLIR: scf.if [[COND]] {
if (a && b && c) {
    U(0, 0, 0) $0;
}

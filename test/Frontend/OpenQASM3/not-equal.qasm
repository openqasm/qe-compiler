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

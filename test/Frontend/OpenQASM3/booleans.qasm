OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

//
// This code is part of Qiskit.
//
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

// MLIR-DAG: %true = arith.constant true
// MLIR-DAG: %false = arith.constant false
// MLIR-DAG: quir.declare_variable @condition_true : i1
// MLIR-DAG: quir.declare_variable @condition_false : i1
// MLIR-DAG: quir.assign_variable @condition_true : i1 = %true
// MLIR-DAG: quir.assign_variable @condition_false : i1 = %false
// AST-PRETTY: DeclarationNode(type=ASTTypeBool, BoolNode(name=condition_true, true))
// AST-PRETTY: DeclarationNode(type=ASTTypeBool, BoolNode(name=condition_false, false))
bool condition_true = true;
bool condition_false = false;

// MLIR: %false_0 = arith.constant false
bool my_bool;

// MLIR: [[CONDITION_TRUE:%.*]] = quir.use_variable @condition_true : i1
// MLIR: scf.if [[CONDITION_TRUE]] {
// MLIR: quir.builtin_U %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>
// AST-PRETTY: condition=IdentifierNode(name=condition_true, bits=8),
if (condition_true) {
  U(1.57079632679, 0.0, 3.14159265359) $0;
}

// MLIR-DAG: [[COND_TRUE:%.*]] = quir.use_variable @condition_true : i1
// MLIR-DAG: [[COND_FALSE:%.*]] = quir.use_variable @condition_false : i1
// MLIR: [[OR:%.*]] = arith.ori [[COND_TRUE]], [[COND_FALSE]] : i1
// MLIR: quir.assign_variable @my_bool : i1 = [[OR]]
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=my_bool, bits=8), right=BinaryOpNode(type=ASTOpTypeLogicalOr, left=IdentifierNode(name=condition_true, bits=8), right=IdentifierNode(name=condition_false, bits=8))
my_bool = condition_true || condition_false;

// MLIR: [[COND_TRUE:%.*]] = quir.use_variable @condition_true : i1
// MLIR: [[CONST_TRUE:%.*]] = arith.constant true
// MLIR: [[NOT:%.*]] = arith.cmpi ne, [[COND_TRUE]], [[CONST_TRUE]] : i1
// MLIR: quir.assign_variable @my_bool : i1 = [[NOT]]
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=my_bool, bits=8), right=UnaryOpNode(type=ASTOpTypeLogicalNot, operand=OperatorNode(op=!, target-identifier=IdentifierNode(name=condition_true, bits=8))
my_bool = !condition_true;

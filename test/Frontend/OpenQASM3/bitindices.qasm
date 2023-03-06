OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

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

// MLIR: module
// MLIR-DAG: func @main

// AST-PRETTY DeclarationNode(type=ASTTypeBitset, CBitNode(name=a, bits=1))
// DeclarationNode(type=ASTTypeBitset, CBitNode(name=b, bits=2, value=10))
// MLIR-DAG: quir.declare_variable @a : !quir.cbit<1>
// MLIR-DAG: quir.declare_variable @b : !quir.cbit<2>
bit a;
bit[2] b = "10";
int c = 5;

// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[CONSTCAST:%[0-9]+]] = "quir.cast"([[CONST]]) : (i32) -> !quir.cbit<1>
// MLIR: quir.assign_variable @a : !quir.cbit<1> = [[CONSTCAST]]
a = 1;

// MLIR: [[A:%.*]] = quir.use_variable @a : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[ACAST:%[0-9]+]] = "quir.cast"([[A]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND:%.*]] = arith.cmpi eq, [[ACAST]], [[CONST]] : i32
// MLIR: scf.if [[COND]] {
if (a == 1) {
// AST-PRETTY: IdentifierRefNode(name=b[0], IdentifierNode(name=b, bits=2), index=0)
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[CONSTCAST:%[0-9]+]] = "quir.cast"([[CONST]]) : (i32) -> i1
// MLIR: quir.assign_cbit_bit @b<2> [0] : i1 = [[CONSTCAST]]
  b[0] = 1;
}
// MLIR: }

// AST-PRETTY: IdentifierRefNode(name=b[1], IdentifierNode(name=b, bits=2), index=1)
// MLIR: [[B:%.*]] = quir.use_variable @b : !quir.cbit<2>
// MLIR: [[BIT:%.*]] = quir.cbit_extractbit([[B]] : !quir.cbit<2>) [1] : i1
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 0 : i32
// MLIR: [[BITCAST:%[0-9]+]] = "quir.cast"([[BIT]]) : (i1) -> i32
// MLIR: [[COND:%.*]] = arith.cmpi eq, [[BITCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND]] {
if (b[1] == 0) {
}
// MLIR: }

bit[2] d;

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32}
qubit $0;

// MLIR: [[MEASUREMENT:%.*]] = quir.measure([[QUBIT0]])
// MLIR: quir.assign_cbit_bit @d<2> [0] : i1 = [[MEASUREMENT]]
d[0] = measure $0;

if (d[0] == 1) {
  d[1] = d[0];
}

d[1] = measure $0;

// AST-PRETTY: operand=OperatorNode(op=!, target-identifier=IdentifierRefNode(name=b[0], IdentifierNode(name=b, bits=2), index=0)
// MLIR: [[B:%.*]] = quir.use_variable @b
// MLIR: [[B0:%.*]] = quir.cbit_extractbit([[B]] : {{.*}}) [0]
// MLIR: [[NOTB0:%.*]] = {{.*}} [[B0]]
// MLIR: quir.assign_cbit_bit @d<2> [0] : i1 = [[NOTB0]]
d[0] = ! b[0];

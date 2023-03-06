OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
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

// MLIR: quir.declare_variable @x : i32
// MLIR: quir.declare_variable @y : i32
// MLIR: quir.declare_variable @a : i3
// MLIR: quir.declare_variable @b : i32

// MLIR: %c100_i32 = arith.constant 100 : i32
// MLIR: quir.assign_variable @x : i32 = %c100_i32
// MLIR: %c100_i32_0 = arith.constant 100 : i32
// MLIR: quir.assign_variable @y : i32 = %c100_i32_0
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=x, value=100, bits=32, signed=1))
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=y, value=100, bits=32, signed=1))
int[32] x = 100;
int[32] y = 100;

// MLIR: %c1_i3 = arith.constant 1 : i3
// MLIR: quir.assign_variable @a : i3 = %c1_i3
// MLIR: %c4_i32 = arith.constant 4 : i32
// MLIR: quir.assign_variable @b : i32 = %c4_i32
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=a, value=1, bits=3, signed=1))
int[3] a = 1;
int b = 4;

qubit $0;
qubit $1;

// MLIR: [[USE_X:%.*]] = quir.use_variable @x : i32
// MLIR: [[USE_Y:%.*]] = quir.use_variable @y : i32
// MLIR: %{{.*}} = arith.cmpi eq, [[USE_X]], [[USE_Y]] : i32
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=x, bits=32), right=IdentifierNode(name=y, bits=32))
if (x == y) {
// MLIR: %c2_i32_1 = arith.constant 2 : i32
// MLIR: [[CAST_A:%.*]] = "quir.cast"(%c2_i32_1) : (i32) -> i3
// MLIR: quir.assign_variable @a : i3 = [[CAST_A]]
// MLIR: %c3_i32_2 = arith.constant 3 : i32
// MLIR: quir.assign_variable @b : i32 = %c3_i32_2
  a = 2;
  b = 3;
  U(0, 0, 0) $0;
}

// MLIR: [[USE_A:%.*]] = quir.use_variable @a : i3
// MLIR: %c2_i32 = arith.constant 2 : i32
// MLIR: [[CAST_A:%.*]] = "quir.cast"([[USE_A]]) : (i3) -> i32
// MLIR: %{{.*}} = arith.cmpi eq, [[CAST_A]], %c2_i32 : i32
if (a == 2) {
  U(0, 0, 0) $1;
}

// MLIR: [[USE_B:%.*]] = quir.use_variable @b : i32
// MLIR: %c3_i32 = arith.constant 3 : i32
// MLIR:  %{{.*}} = arith.cmpi eq, [[USE_B]], %c3_i32 : i32
if (b == 3) {
  U(0, 0, 0) $0;
}

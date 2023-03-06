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


// MLIR-DAG: quir.declare_variable @a : i1
// MLIR-DAG: quir.declare_variable @j : !quir.cbit<1>

// Angle
// MLIR: %{{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<3>>
angle[3] c = 0;
// MLIR: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;
// MLIR: %c1_i32{{.*}} = arith.constant 1 : i32
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=c, bits=3), right=IntNode(signed=true, value=1, bits=32))
c = 1;
// MLIR: %[[CAST_RESULT:.*]] = "quir.cast"(%{{.*}}) : (i32) -> !quir.angle<3>
// MLIR:       quir.assign_variable @c : !quir.angle<3> = %[[CAST_RESULT]]
// MLIR:       %[[USE_RESULT:.*]] = quir.use_variable @c : !quir.angle<3>
// MLIR: %[[CAST_RESULT1:.*]] = "quir.cast"(%[[USE_RESULT]]) : (!quir.angle<3>) -> i1
// MLIR: scf.if %[[CAST_RESULT1]] {
if (c) {
    U(0,0,0) $0;
}

// Boolean
// MLIR: %[[TRUE:.*]] = arith.constant true
// MLIR: quir.assign_variable @a : i1 = %[[TRUE]]
bool a = true;
// MLIR: %[[FALSE:.*]] = arith.constant false
// MLIR: quir.assign_variable @a : i1 = %[[FALSE]]
// ASR-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=a, bits=8), right=BoolNode(name=bool, false))
a = false;
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
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
// MLIR: %[[CAST_RESULT4:.*]] = "quir.cast"(%{{.*}}) : (i32) -> i1
// MLIR: scf.if %[[CAST_RESULT4]] {
if (x) {
    U(0, 0, 0) $0;
}

OPENQASM 3.0;
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
bool a = true;
bool b = false;

// Test for ==
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi eq, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a == b){
    U(0, 0, 0) $0;
}

// Test for <
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi slt, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a < b){
    U(0, 0, 0) $0;
}

// Test for >
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi sgt, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a > b){
    U(0, 0, 0) $0;
}

// Test for >=
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi sge, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a >= b){
    U(0, 0, 0) $0;
}

// Test for <=
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi sle, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a <= b){
    U(0, 0, 0) $0;
}

// Test for !=
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi ne, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a != b){
    U(0, 0, 0) $0;
}

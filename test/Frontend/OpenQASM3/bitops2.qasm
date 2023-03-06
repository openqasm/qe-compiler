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
// MLIR: func @main

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer
qubit $0;

gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}

x $0;

bit a;
bit b = 0;

a = measure $0; // expected "1"

if (a == 1) {
  // expected branch
  b = 1;
}

if (b == 0) {
  // expected not to happen
  x $0;
}

b = measure $0; // expected "1"

// on hardware, expect to measure "11"

// previously, this code failed to compile while lowering the comparison in
// second if condition

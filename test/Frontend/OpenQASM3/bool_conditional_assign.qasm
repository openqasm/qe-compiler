OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR
//
// Test that bool variables are handled correctly across control flow.

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

// MLIR: module

// AST: <Declaration>
// AST: <Identifier>$0</Identifier>
// AST: <Type>ASTTypeQubitContainer</Type>
// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;
gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}
x $0;

// AST: <Declaration>
// AST: <Identifier>a</Identifier>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=a, bits=1))
// MLIR: quir.declare_variable @a : !quir.cbit<1>
bit a;

// AST: <Declaration>
// AST: <Identifier>b</Identifier>
// AST-PRETTY: DeclarationNode(type=ASTTypeBool, BoolNode(name=b, false))
bool b = false;

a = measure $0; // expected "1"
if (a == 1) { // expected to execute
  b = true;
}

if ( ! b ) { // expected not to execute since b was set to true
  x $0;
}

a = measure $0; // expected "1"

// on hardware, we expect to measure "11" (in the overwhelming majority of shots)

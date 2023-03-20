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
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.


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

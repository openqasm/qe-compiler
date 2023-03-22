OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

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

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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

// AST-PRETTY: StretchStatementNode(StretchNode(name=a))
// AST-PRETTY: StretchStatementNode(StretchNode(name=b))
// MLIR: {{.*}} = oq3.declare_stretch : !quir.stretch
// MLIR: {{.*}} = oq3.declare_stretch : !quir.stretch
stretch a;
stretch b;

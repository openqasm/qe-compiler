OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false| FileCheck %s --match-full-lines --check-prefixes MLIR
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s --match-full-lines --check-prefixes MLIR

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

int i = 15;
qubit $0;
// MLIR: quir.switch %{{.*}}{
// MLIR: }[0 : {
// MLIR: }]
switch (i) {
    case 0: {
    }
    break;
}

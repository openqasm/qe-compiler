OPENQASM 3.0;
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

// MLIR: func @g(%arg0: !quir.qubit<1>, %arg1: !quir.angle<{{.*}}>) {
// MLIR: quir.builtin_U %arg0, {{.*}}, {{.*}}, %arg1 : !quir.qubit<1>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>
gate g (theta) q {
    U(0.0, 0.0, theta) q;
}

// MLIR: func @g3(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>, %arg2: !quir.angle<{{.*}}>, %arg3: !quir.angle<{{.*}}>, %arg4: !quir.angle<{{.*}}>) {
// MLIR: quir.builtin_U %arg0, %arg2, %arg4, %arg3 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: quir.builtin_U %arg1, %arg2, %arg4, %arg3 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: quir.builtin_CX %arg0, %arg1 : !quir.qubit<1>, !quir.qubit<1>
gate g3 (theta, lambda, phi) qa, qb {
    U(theta, phi, lambda) qa;
    U(theta, phi, lambda) qb;
    CX qa, qb;
}

// MLIR: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
qubit $2;
qubit $3;

// MLIR: quir.call_gate @g([[QUBIT2]], {{.*}}) : (!quir.qubit<1>, !quir.angle<{{.*}}>) -> ()
g (3.14) $2;
// MLIR: quir.call_gate @g3([[QUBIT2]], [[QUBIT3]], {{.*}}, {{.*}}, {{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>) -> ()
g3 (3.14, 1.2, 0.2) $2, $3;

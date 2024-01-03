OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

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

// MLIR-NO-CIRCUITS: func.func @g([[QUBIT:%.*]]: !quir.qubit<1>, [[ANGLE:%.*]]: !quir.angle<{{.*}}>) {
// MLIR-CIRCUITS:quir.circuit @circuit_0([[ANGLE:%.*]]: !quir.angle<64>, [[QUBIT:%.*]]: !quir.qubit<1>) {
// MLIR: quir.builtin_U [[QUBIT]], {{.*}}, {{.*}}, [[ANGLE]] : !quir.qubit<1>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>
gate g (theta) q {
    U(0.0, 0.0, theta) q;
}

// MLIR-NO-CIRCUITS: func.func @g3([[QUBIT0:%.*]]: !quir.qubit<1>, [[QUBIT1:%.*]]: !quir.qubit<1>, [[ANGLE0:%.*]]: !quir.angle<{{.*}}>, [[ANGLE1:%.*]]: !quir.angle<{{.*}}>, [[ANGLE2:%.*]]: !quir.angle<{{.*}}>) {
// MLIR-CIRCUITS: quir.circuit @circuit_1({{.*}}: !quir.{{.*}}, {{.*}}: !quir.{{.*}}, {{.*}}: !quir.{{.*}}, {{.*}}: !quir.{{.*}}, {{.*}}: !quir.{{.*}}) {
// NOTE can not enforce parameter ordering on the builtin_U because the order of the quir.circuit parameters changes when tested with github actions
// MLIR: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: quir.builtin_CX {{.*}}, {{.*}} : !quir.qubit<1>, !quir.qubit<1>
gate g3 (theta, lambda, phi) qa, qb {
    U(theta, phi, lambda) qa;
    U(theta, phi, lambda) qb;
    CX qa, qb;
}

// MLIR-CIRCUITS: quir.circuit @circuit_2(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
// MLIR-CIRCUITS: quir.call_gate @g(%arg1, %angle) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// MLIR-CIRCUITS: quir.call_gate @g3(%arg1, %arg0, %angle_0, %angle_1, %angle_2) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>) -> ()

// MLIR: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
qubit $2;
qubit $3;

// MLIR-NO-CIRCUITS: quir.call_gate @g([[QUBIT2]], {{.*}}) : (!quir.qubit<1>, !quir.angle<{{.*}}>) -> ()
g (3.14) $2;
// MLIR-NO-CIRCUITS: quir.call_gate @g3([[QUBIT2]], [[QUBIT3]], {{.*}}, {{.*}}, {{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>, !quir.angle<{{.*}}>) -> ()
// MLIR-CIRCUITS: quir.call_circuit @circuit_2(%1, %0) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
g3 (3.14, 1.2, 0.2) $2, $3;

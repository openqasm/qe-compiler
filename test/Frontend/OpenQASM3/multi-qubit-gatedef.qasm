OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm=false| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

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

// MLIR: func.func @g([[QUBIT0:%.*]]: !quir.qubit<1>, [[QUBIT1:%.*]]: !quir.qubit<1>) {
// MLIR-CIRCUITS: quir.circuit @circuit_0([[QUBIT1:%.*]]: !quir.qubit<1>, [[QUBIT0:%.*]]: !quir.qubit<1>) {
gate g qa, qb {
    // MLIR: quir.builtin_U [[QUBIT0]]{{.*}}
    // MLIR: quir.builtin_U [[QUBIT1]]{{.*}}
    U(1.57079632679, 0.0, 3.14159265359) qa;
    U(1.57079632679, 0.0, 3.14159265359) qb;
}

// MLIR: func.func @g4(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>, %arg2: !quir.qubit<1>) {
// MLIR-CIRCUITS: quir.circuit @circuit_1([[QUBIT0:%.*]]: !quir.qubit<1>, [[QUBIT1:%.*]]: !quir.qubit<1>, [[QUBIT2:%.*]]: !quir.qubit<1>) {
gate g4 qa, qb, qc {
    U(1.57079632679, 0.0, 3.14159265359) qa;
    U(0.0, 0.0, 3.14159265359) qb;
    U(1.57079632679, 0.0, 0.0) qc;
}

qubit $2;
qubit $3;
qubit $4;

// MLIR-NO-CIRCUITS: quir.call_gate @g(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
// MLIR-NO-CIRCUITS: quir.call_gate @g4(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()
// MLIR-CIRCUITS: quir.call_circuit @circuit_2({{.*}}, {{.*}}, {{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()
g $2, $3;
g4 $2, $3, $4;

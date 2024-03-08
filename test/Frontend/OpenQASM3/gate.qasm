OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm=false| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// MLIR: func.func @h(%arg0: !quir.qubit<1>) {
// MLIR-NO-CIRCUITS: %angle = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
// MLIR-NO-CIRCUITS: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.call_circuit @circuit_0(%arg0) : (!quir.qubit<1>) -> ()
gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

// MLIR-CIRCUITS: quir.circuit @circuit_0(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: %angle = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_0 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_1 = quir.constant #quir.angle<3.1415926535900001> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.return
// MLIR-CIRCUITS: quir.circuit @circuit_1(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: %angle = quir.constant #quir.angle<3.140000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_0 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_1 = quir.constant #quir.angle<3.140000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.call_gate @h(%arg0) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS: quir.return

// MLIR: func.func @main() -> i32 {

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;

// MLIR-NO-CIRCUITS: quir.builtin_U [[QUBIT0]], %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
U(3.14, 0, 3.14) $0;
// MLIR-NO-CIRCUITS: quir.call_gate @h([[QUBIT0]]) : (!quir.qubit<1>) -> ()
h $0;
// MLIR-CIRCUITS: quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> ()

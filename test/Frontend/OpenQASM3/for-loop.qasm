OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
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

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;

// For loop.
// MLIR: %c0_0 = arith.constant 0 : index
// MLIR: %c5 = arith.constant 5 : index
// MLIR: %c1_1 = arith.constant 1 : index
// MLIR: scf.for %arg1 = %c0_0 to %c5 step %c1_1 {
// AST-PRETTY: ForStatementNode(start=0, end=4,
for i in [0 : 4] {
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<3.1415926535900001> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // MLIR-CIRCUITS: quir.call_circuit @circuit_0(%0) : (!quir.qubit<1>) -> ()
    // AST-PRETTY: statements=
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[$0])
    // AST-PRETTY: )
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

// MLIR: %c0_2 = arith.constant 0 : index
// MLIR: %c5_3 = arith.constant 5 : index
// MLIR: %c1_4 = arith.constant 1 : index
// MLIR: scf.for %arg1 = %c0_2 to %c5_3 step %c1_4 {
// AST-PRETTY: ForStatementNode(start=0, stepping=1, end=4,
for i in [0 : 1 : 4] {
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<3.1415926535900001> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // MLIR-CIRCUITS: quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> ()
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

// MLIR: %c-1 = arith.constant -1 : index
// MLIR: %c1001 = arith.constant 1001 : index
// MLIR: %c10 = arith.constant 10 : index
// MLIR: scf.for %arg1 = %c-1 to %c1001 step %c10 {
// AST-PRETTY: ForStatementNode(start=-1, stepping=10, end=1000,
for i in [-1 : 10 : 1000] {
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<3.1415926535900001> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // MLIR-CIRCUITS: quir.call_circuit @circuit_2(%0) : (!quir.qubit<1>) -> ()
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm=false | FileCheck %s --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm | FileCheck %s --check-prefixes MLIR,MLIR-CIRCUITS

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

// Regression test for an issue where successive gate calls used incorrect angle
// parameters.

// MLIR-CIRCUITS: quir.circuit @circuit_0([[ARG0:%.*]]: !quir.qubit<1>) {
// MLIR-CIRCUITS: [[ANGLE_1_1:%.*]] = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: [[ANGLE_1_2:%.*]] = quir.constant #quir.angle<1.000000e-01> : !quir.angle<64>
// MLIR-CIRCUITS: [[ANGLE_1_3:%.*]] = quir.constant #quir.angle<2.000000e-01> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U [[ARG0]], [[ANGLE_1_1]], [[ANGLE_1_2]], [[ANGLE_1_3]]
// MLIR-CIRCUITS: [[ANGLE_2_1:%.*]] = quir.constant #quir.angle<1.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: [[ANGLE_2_2:%.*]] = quir.constant #quir.angle<1.100000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: [[ANGLE_2_3:%.*]] = quir.constant #quir.angle<1.250000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U [[ARG0]], [[ANGLE_2_1]], [[ANGLE_2_2]], [[ANGLE_2_3]]
// MLIR-CIRCUITS: quir.return

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 :
qubit $0;

// AST-PRETTY: UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.1, bits=64), AngleNode(value=0.2, bits=64)], qubits=[], qcparams=[$0])
// MLIR-NO-CIRCUITS: [[ANGLE_1_1:%.*]] = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-NO-CIRCUITS: [[ANGLE_1_2:%.*]] = quir.constant #quir.angle<1.000000e-01> : !quir.angle<64>
// MLIR-NO-CIRCUITS: [[ANGLE_1_3:%.*]] = quir.constant #quir.angle<2.000000e-01> : !quir.angle<64>
// MLIR-NO-CIRCUITS: quir.builtin_U [[QUBIT0]], [[ANGLE_1_1]], [[ANGLE_1_2]], [[ANGLE_1_3]]
U(0.0, 0.1, 0.2) $0;

// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.0, bits=64), AngleNode(value=1.1, bits=64), AngleNode(value=1.25, bits=64)], qubits=[], qcparams=[$0])
// MLIR-NO-CIRCUITS: [[ANGLE_2_1:%.*]] = quir.constant #quir.angle<1.000000e+00> : !quir.angle<64>
// MLIR-NO-CIRCUITS: [[ANGLE_2_2:%.*]] = quir.constant #quir.angle<1.100000e+00> : !quir.angle<64>
// MLIR-NO-CIRCUITS: [[ANGLE_2_3:%.*]] = quir.constant #quir.angle<1.250000e+00> : !quir.angle<64>
// MLIR-NO-CIRCUITS: quir.builtin_U [[QUBIT0]], [[ANGLE_2_1]], [[ANGLE_2_2]], [[ANGLE_2_3]]
// MLIR-CIRCUITS: quir.call_circuit @circuit_0(%0) : (!quir.qubit<1>) -> ()
U(1.0, 1.1, 1.25) $0;

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

// Regression test for an issue where successive gate calls used incorrect angle
// parameters.

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 :
qubit $0;

// AST-PRETTY: UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.1, bits=64), AngleNode(value=0.2, bits=64)], qubits=[], qcparams=[$0])
// MLIR: [[ANGLE_1_1:%.*]] = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_1_2:%.*]] = quir.constant #quir.angle<1.000000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_3:%.*]] = quir.constant #quir.angle<2.000000e-01 : !quir.angle<64>>
// MLIR: quir.builtin_U [[QUBIT0]], [[ANGLE_1_1]], [[ANGLE_1_2]], [[ANGLE_1_3]]
U(0.0, 0.1, 0.2) $0;

// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.0, bits=64), AngleNode(value=1.1, bits=64), AngleNode(value=1.25, bits=64)], qubits=[], qcparams=[$0])
// MLIR: [[ANGLE_2_1:%.*]] = quir.constant #quir.angle<1.000000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_2:%.*]] = quir.constant #quir.angle<1.100000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_3:%.*]] = quir.constant #quir.angle<1.250000e+00 : !quir.angle<64>>
// MLIR: quir.builtin_U [[QUBIT0]], [[ANGLE_2_1]], [[ANGLE_2_2]], [[ANGLE_2_3]]
U(1.0, 1.1, 1.25) $0;

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

// Regression test for an issue where successive rz gate calls used incorrect
// angle parameters.

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 :
qubit $0;

gate rz(theta) q {}

// AST-PRETTY: GateGenericNode(name=rz, params=[AngleNode(value=1.0, bits=64)], qubits=[], qcparams=[$0])
// MLIR: [[ANGLE1:%.*]] = quir.constant #quir.angle<1.000000e+00 : !quir.angle<64>>
// MLIR: quir.call_gate @rz([[QUBIT0]], [[ANGLE1]])
rz(1.0) $0;

// AST-PRETTY: GateGenericNode(name=rz, params=[AngleNode(value=2.0, bits=64)], qubits=[], qcparams=[$0])
// MLIR: [[ANGLE2:%.*]] = quir.constant #quir.angle<2.000000e+00 : !quir.angle<64>>
// MLIR: quir.call_gate @rz([[QUBIT0]], [[ANGLE2]])
rz(2.0) $0;

// AST-PRETTY: GateGenericNode(name=rz, params=[AngleNode(value=3.0, bits=64)], qubits=[], qcparams=[$0])
// MLIR: [[ANGLE3:%.*]] = quir.constant #quir.angle<3.000000e+00 : !quir.angle<64>>
// MLIR: quir.call_gate @rz([[QUBIT0]], [[ANGLE3]])
rz(3.0) $0;

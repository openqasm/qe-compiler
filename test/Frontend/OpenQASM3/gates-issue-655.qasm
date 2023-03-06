OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

// (C) Copyright IBM 2023.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Regression test for an issue where successive gate calls used incorrect angle
// parameters.

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 :
qubit $0;

gate star(alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota) q { }

// AST-PRETTY: GateGenericNode(name=star, params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.1, bits=64), AngleNode(value=0.2, bits=64), AngleNode(value=0.25, bits=64), AngleNode(value=0.4, bits=64), AngleNode(value=0.5, bits=64), AngleNode(value=0.75, bits=64), AngleNode(value=0.8, bits=64), AngleNode(value=0.9, bits=64)], qubits=[], qcparams=[$0])
// MLIR: [[ANGLE_1_1:%.*]] = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_1_2:%.*]] = quir.constant #quir.angle<1.000000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_3:%.*]] = quir.constant #quir.angle<2.000000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_4:%.*]] = quir.constant #quir.angle<2.500000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_5:%.*]] = quir.constant #quir.angle<4.000000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_6:%.*]] = quir.constant #quir.angle<5.000000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_7:%.*]] = quir.constant #quir.angle<7.500000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_8:%.*]] = quir.constant #quir.angle<8.000000e-01 : !quir.angle<64>>
// MLIR: [[ANGLE_1_9:%.*]] = quir.constant #quir.angle<9.000000e-01 : !quir.angle<64>>
// MLIR: quir.call_gate @star([[QUBIT0]], [[ANGLE_1_1]], [[ANGLE_1_2]], [[ANGLE_1_3]], [[ANGLE_1_4]], [[ANGLE_1_5]], [[ANGLE_1_6]], [[ANGLE_1_7]], [[ANGLE_1_8]], [[ANGLE_1_9]])
star(0.0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9) $0;

// AST-PRETTY: GateGenericNode(name=star, params=[AngleNode(value=1.0, bits=64), AngleNode(value=1.1, bits=64), AngleNode(value=1.25, bits=64), AngleNode(value=1.5, bits=64), AngleNode(value=1.75, bits=64), AngleNode(value=1.8, bits=64), AngleNode(value=2.0, bits=64), AngleNode(value=2.25, bits=64), AngleNode(value=2.5, bits=64)], qubits=[], qcparams=[$0]
// MLIR: [[ANGLE_2_1:%.*]] = quir.constant #quir.angle<1.000000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_2:%.*]] = quir.constant #quir.angle<1.100000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_3:%.*]] = quir.constant #quir.angle<1.250000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_4:%.*]] = quir.constant #quir.angle<1.500000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_5:%.*]] = quir.constant #quir.angle<1.750000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_6:%.*]] = quir.constant #quir.angle<1.800000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_7:%.*]] = quir.constant #quir.angle<2.000000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_8:%.*]] = quir.constant #quir.angle<2.250000e+00 : !quir.angle<64>>
// MLIR: [[ANGLE_2_9:%.*]] = quir.constant #quir.angle<2.500000e+00 : !quir.angle<64>>
// MLIR: quir.call_gate @star([[QUBIT0]], [[ANGLE_2_1]], [[ANGLE_2_2]], [[ANGLE_2_3]], [[ANGLE_2_4]], [[ANGLE_2_5]], [[ANGLE_2_6]], [[ANGLE_2_7]], [[ANGLE_2_8]], [[ANGLE_2_9]])
star(1.0, 1.1, 1.25, 1.5, 1.75, 1.8, 2.0, 2.25, 2.5) $0;

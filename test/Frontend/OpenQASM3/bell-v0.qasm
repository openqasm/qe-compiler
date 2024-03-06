OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm=false| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

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

// Qiskit will provide physical qubits

// AST: <Declaration>
// AST: <Identifier>$0</Identifier>
// AST: <Type>ASTTypeQubitContainer</Type>
// AST: <IsTypeDeclaration>false</IsTypeDeclaration>
// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$1:0, bits=1)))
// MLIR-NO-CIRCUITS-DAG: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR-NO-CIRCUITS-DAG: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
qubit $0;
qubit $1;

// define classical bits

// AST: <Declaration>
// AST: <Identifier>c0</Identifier>
// AST: <Type>ASTTypeBitset</Type>
// AST: <Identifier>
// AST: <Name>c0</Name>
// AST: <Bits>1</Bits>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c0, bits=1))
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c1, bits=1))

// MLIR-DAG: oq3.declare_variable @c0 : !quir.cbit<1>
// MLIR-DAG: oq3.declare_variable @c1 : !quir.cbit<1>
bit c0;
bit c1;

// AST: <Gate>
// AST: <Name>U</Name>
// AST: <Gate>
// AST: <Name>cx</Name>
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[$0])
// AST-PRETTY: GateDeclarationNode(name=cx, params=[], qubits=[QubitNode(name=ast-gate-qubit-param-control-0, bits=0), QubitNode(name=ast-gate-qubit-param-target-1, bits=0)], qcparams=[control, target])
// MLIR-CIRCUITS: quir.circuit @circuit_0([[ARG0:%[A-Za-z0-9]+]]: !quir.qubit<1>) {
// MLIR: [[a0:%angle[_0-9]*]] = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
// MLIR-NO-CIRCUITS: quir.builtin_U [[QUBIT0]], [[a0]], {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-NO-CIRCUITS: quir.builtin_CX [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>
// MLIR-CIRCUITS: [[a1:%angle[_0-9]*]] = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: [[a2:%angle[_0-9]*]] = quir.constant #quir.angle<3.1415926535900001> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U [[ARG0]], [[a0]], [[a1]], [[a2]] : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.circuit @circuit_1([[ARG0:%[A-Za-z0-9]+]]: !quir.qubit<1>, [[ARG1:%[A-Za-z0-9]+]]: !quir.qubit<1>) -> i1 {
// MLIR-CIRCUITS: quir.builtin_CX [[ARG1]], [[ARG0]] : !quir.qubit<1>, !quir.qubit<1>
// MLIR-CIRCUITS: [[m0:%[_0-9]*]] = quir.measure([[ARG1]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: quir.return [[m0]] : i1
// MLIR-CIRCUITS: quir.circuit @circuit_2([[ARG0:%[A-Za-z0-9]+]]: !quir.qubit<1>) -> i1 {
// MLIR-CIRCUITS: [[m1:%[_0-9]*]] = quir.measure([[ARG0]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: quir.return [[m1]] : i1
U(1.57079632679, 0.0, 3.14159265359) $0;
gate cx control, target {}
cx $0, $1;

// AST: <MeasureNode>
// AST: <Target>
// AST: <QubitContainer>
// AST: <Identifier>$0</Identifier>
// AST: <Size>1</Size>
// AST: <Result>
// AST: <CBit>
// AST: <Identifier>
// AST: <Name>c0</Name>
// AST: <Bits>1</Bits>
// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=c0, bits=1))
// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=c1, bits=1))
// MLIR-CIRCUITS-DAG: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR-CIRCUITS-DAG: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// MLIR-CIRCUITS: quir.call_circuit @circuit_0([[QUBIT0]]) : (!quir.qubit<1>) -> ()
// MLIR-NO-CIRCUITS: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE0:%.]] = quir.call_circuit @circuit_1([[QUBIT1]], [[QUBIT0]]) : (!quir.qubit<1>, !quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @c0<1> [0] : i1 = [[MEASURE0]]
// MLIR-NO-CIRCUITS: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE1:%.]] = quir.call_circuit @circuit_2([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @c1<1> [0] : i1 = [[MEASURE1]]
measure $0 -> c0;
measure $1 -> c1;

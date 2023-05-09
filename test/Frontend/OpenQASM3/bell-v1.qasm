OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false | FileCheck %s --match-full-lines  --check-prefixes MLIR,MLIR-NO-CIRCUITS
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

// MLIR: oq3.declare_variable @c : !quir.cbit<2>

// Define a hadamard gate

// AST: <GateDeclarationNode>
// AST: <HadamardGate>
// AST: <Gate>
// AST: <Name>h</Name>
// AST-PRETTY: GateDeclarationNode(name=h, params=[], qubits=[QubitNode(name=ast-gate-qubit-param-q-0, bits=0)], qcparams=[q],
// AST-PRETTY: ops=[
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
// AST-PRETTY: ,
// AST-PRETTY: ]
// AST-PRETTY: )
// MLIR: func @h(%arg0: !quir.qubit<1>) {
// MLIR-NO-CIRCUITS: [[a0:%angle[_0-9]*]] = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
// MLIR-NO-CIRCUITS: quir.builtin_U %arg0, [[a0]], {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.call_circuit @circuit_0(%arg0) : (!quir.qubit<1>) -> ()
// MLIR: return
// MLIR-CIRCUITS: quir.circuit @circuit_0(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: [[a0:%angle[_0-9]*]] = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
// MLIR-CIRCUITS: [[a1:%angle[_0-9]*]] = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR-CIRCUITS: [[a2:%angle[_0-9]*]] = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
// MLIR-CIRCUITS: quir.builtin_U %arg0, [[a0]], [[a1]], [[a2]] : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.return
// MLIR-CIRCUITS: quir.circuit @circuit_1(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) -> (i1, i1) {
// MLIR-CIRCUITS: quir.call_gate @h(%arg1) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS: quir.builtin_CX %arg1, %arg0 : !quir.qubit<1>, !quir.qubit<1>
// MLIR-CIRCUITS: %0 = quir.measure(%arg1) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: %1 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: quir.return %0, %1 : i1, i1

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

gate cx control, target { }

// MLIR: func @main() -> i32 {
// Qiskit will provide physical qubits
// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
qubit $0;
qubit $1;

// Array of classical bits is fine
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c, bits=2))

bit[2] c;

// AST-PRETTY: HGateOpNode(params=[], qubits=[], qcparams=[$0],
// AST-PRETTY: ops=[
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
// MLIR-NO-CIRCUITS: quir.call_gate @h([[QUBIT0]]) : (!quir.qubit<1>) -> ()
// MLIR-NO-CIRCUITS: quir.builtin_CX [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>
h $0;
cx $0, $1;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=c, bits=2)[index=0])
// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=c, bits=2)[index=1])
// MLIR-NO-CIRCUITS: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR-NO-CIRCUITS: oq3.cbit_assign_bit @c<2> [0] : i1 = [[MEASURE0]]
// MLIR-NO-CIRCUITS: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR-NO-CIRCUITS: oq3.cbit_assign_bit @c<2> [1] : i1 = [[MEASURE1]]
// MLIR-CIRCUITS: [[MEASURE0:%.]]:[[LEN:.]] = quir.call_circuit @circuit_1([[QUBIT1]], [[QUBIT0]]) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
// MLIR-CIRCUITS-DAG: oq3.cbit_assign_bit @c<2> [0] : i1 = [[MEASURE0]]#0
// MLIR-CIRCUITS-DAG: oq3.cbit_assign_bit @c<2> [1] : i1 = [[MEASURE0]]#1
measure $0 -> c[0];
measure $1 -> c[1];

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

// MLIR-CIRCUITS: quir.circuit @circuit_1(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: %angle = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_0 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_1 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.return

// MLIR-CIRCUITS: quir.circuit @circuit_2(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: %angle = quir.constant #quir.angle<3.1415926000000001> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_0 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-CIRCUITS: %angle_1 = quir.constant #quir.angle<3.1415926000000001> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.return

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;

// AST-PRETTY: DeclarationNode(type=ASTTypeBool, BoolNode(name=b, false))
// MLIR: %false = arith.constant false
bool b = false;

// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=UnaryOpNode(type=ASTOpTypeLogicalNot, operand=OperatorNode(op=!, target-identifier=IdentifierNode(name=b, bits=8))
// AST-PRETTY: statements=
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=0.0, bits=64)], qubits=[], qcparams=[$0])
// AST-PRETTY: )
// MLIR-DAG: %true = arith.constant true
// MLIR: {{.*}} = arith.cmpi ne, {{.*}}, %true : i1
// MLIR: scf.if {{.*}} {
if (!b) {
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-CIRCUITS: quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> ()
    U(0, 0, 0) $0;
}
// AST-PRETTY: ElseStatementNode(
// AST-PRETTY: statements=
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=3.1415926, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.1415926, bits=64)], qubits=[], qcparams=[$0])
// AST-PRETTY: )
// MLIR: } else {
else {
    //MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<3.1415926000000001> : !quir.angle<64>
    //MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    //MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.angle<3.1415926000000001> : !quir.angle<64>
    // MLIR-CIRCUITS: quir.call_circuit @circuit_2(%0) : (!quir.qubit<1>) -> ()
    U(3.1415926, 0, 3.1415926) $0;
}

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
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

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

qubit $0;
int i = 1;
int j = 0;
bit is_excited;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=i, bits=32), right=IntNode(signed=true, value=0, bits=32))
// AST-PRETTY: ,
// AST-PRETTY: statements=
// MLIR: scf.while : () -> () {
// MLIR:     %2 = oq3.variable_load @i : i32
// MLIR:     %c0_i32_1 = arith.constant 0 : i32
// MLIR:     %3 = arith.cmpi ne, %2, %c0_i32_1 : i32
// MLIR:     scf.condition(%3)
// MLIR: } do {
while (i != 0) {
    // AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=j, bits=32), right=IntNode(signed=true, value=0, bits=32))
    // AST-PRETTY: ,
    // MLIR:     scf.while : () -> () {
    // MLIR:         %3 = oq3.variable_load @j : i32
    // MLIR:         [[CONSTANT0:%c0_i32_.]] = arith.constant 0 : i32
    // MLIR:         %4 = arith.cmpi ne, %3, [[CONSTANT0]] : i32
    // MLIR:         scf.condition(%4)
    // MLIR:     } do {
    while (j != 0) {
        // AST-PRETTY: statements=
        // AST-PRETTY: HGateOpNode(params=[], qubits=[], qcparams=[$0],
        // AST-PRETTY: ops=[
        // AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
        // AST-PRETTY: ,
        // AST-PRETTY: ]
        // AST-PRETTY: )
        // MLIR-NO-CIRCUITS:         quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
        // MLIR-CIRCUITS: quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> ()
        // MLIR:         [[CONSTANT1:%c0_i32_.]] = arith.constant 0 : i32
        // MLIR:         oq3.variable_assign @j : i32 = [[CONSTANT1]]
        // MLIR:         scf.yield
        // MLIR:     }
        h $0;
        // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=j, bits=32), right=IntNode(signed=true, value=0, bits=32))
        j = 0;
    }
    // MLIR-NO-CIRCUITS:     %angle = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS:     %angle_1 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS:     %angle_2 = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
    // MLIR-NO-CIRCUITS:     quir.builtin_U %0, %angle, %angle_1, %angle_2 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    U (0, 0, 0) $0;
    // MLIR-NO-CIRCUITS:     %2 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // MLIR-CIRCUITS: %2 = quir.call_circuit @circuit_2(%0) : (!quir.qubit<1>) -> i1
    // MLIR:     oq3.cbit_assign_bit @is_excited<1> [0] : i1 = %2
    is_excited = measure $0;
    // MLIR:     [[CONSTANT2:%c0_i32_.]] = arith.constant 0 : i32
    // MLIR:     oq3.variable_assign @i : i32 = [[CONSTANT2]]
    // MLIR:     scf.yield
    // MLIR: }
    i = 0;
}

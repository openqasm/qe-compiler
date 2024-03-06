OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm=false | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
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
int n = 1;

bit is_excited;

// MLIR-CIRCUITS: func.func @h(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: quir.call_circuit @circuit_0(%arg0) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS: return

// MLIR-CIRCUITS: quir.circuit @circuit_0(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: %angle = quir.constant #quir.angle<1.57079632679> : !quir.angle<64>
// MLIR-CIRCUITS: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS: quir.return

// MLIR-CIRCUITS: quir.circuit @circuit_1(%arg0: !quir.qubit<1>) -> i1 {
// MLIR-CIRCUITS: quir.call_gate @h(%arg0) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS: %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: quir.return %0 : i1

// MLIR-CIRCUITS: quir.circuit @circuit_2(%arg0: !quir.qubit<1>) {
// MLIR-CIRCUITS: quir.call_gate @h(%arg0) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS: quir.return

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=n, bits=32), right=IntNode(signed=true, value=0, bits=32))
// MLIR: scf.while : () -> () {
// MLIR:     %2 = oq3.variable_load @n : i32
// MLIR:     %c0_i32_0 = arith.constant 0 : i32
// MLIR:     %3 = arith.cmpi ne, %2, %c0_i32_0 : i32
// MLIR:     scf.condition(%3)
// MLIR: } do {
while (n != 0) {
    // AST-PRETTY: statements=
    // AST-PRETTY: HGateOpNode(params=[], qubits=[], qcparams=[$0],
    // AST-PRETTY: ops=[
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
    // MLIR-NO-CIRCUITS: quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
    h $0;
    // MLIR-NO-CIRCUITS: %2 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // MLIR-CIRCUITS: %2 = quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> i1
    // MLIR: oq3.cbit_assign_bit @is_excited<1> [0] : i1 = %2
    // MLIR: %3 = oq3.variable_load @is_excited : !quir.cbit<1>
    // MLIR: %4 = "oq3.cast"(%3) : (!quir.cbit<1>) -> i1
    is_excited = measure $0;
    // MLIR: scf.if %4 {
    if (is_excited) {
        // MLIR-NO-CIRCUITS:     quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
        // MLIR-CIRCUITS: quir.call_circuit @circuit_2(%0) : (!quir.qubit<1>) -> ()
        // MLIR: }
        h $0;
    }
    // error: Binary operation ASTOpTypeSub not supported yet.
    // n = n - 1;
    // MLIR: [[REG:.*]] = arith.constant 0 : i32
    // MLIR: oq3.variable_assign @n : i32 = [[REG]]
    n = 0;  // workaround for n = n - 1
    // MLIR: scf.yield
}
// AST-PRETTY: )

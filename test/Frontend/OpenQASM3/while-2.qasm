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

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

qubit $0;
int n = 2;
bit is_excited;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=n, bits=32), right=IntNode(signed=true, value=0, bits=32))
// MLIR: scf.while : () -> () {
// MLIR:     %2 = oq3.variable_load @n : i32
// MLIR:     %c0_i32_0 = arith.constant 0 : i32
// MLIR:     %3 = arith.cmpi ne, %2, %c0_i32_0 : i32
// MLIR:     scf.condition(%3)
// MLIR: } do {
while (n != 0) {
    // MLIR: %2 = oq3.variable_load @n : i32
    // MLIR: %c2_i32_0 = arith.constant 2 : i32
    // MLIR: %3 = arith.cmpi eq, %2, %c2_i32_0 : i32
    // MLIR: scf.if %3 {
    if (n == 2) {
        // MLIR-NO-CIRCUITS: quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
        // MLIR-CIRCUITS: quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> ()
        // MLIR: %c1_i32 = arith.constant 1 : i32
        // MLIR: oq3.variable_assign @n : i32 = %c1_i32
        h $0;
        n = 1;
    // MLIR: } else {
    } else {
        // MLIR-NO-CIRCUITS: quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
        // MLIR-NO-CIRCUITS: %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
        // MLIR-CIRCUITS: %4 = quir.call_circuit @circuit_2(%0) : (!quir.qubit<1>) -> i1
        // MLIR: oq3.cbit_assign_bit @is_excited<1> [0] : i1 = %4
        // MLIR: %c0_i32_2 = arith.constant 0 : i32
        // MLIR: oq3.variable_assign @n : i32 = %c0_i32_2
        h $0;
        is_excited = measure $0;
        n = 0;
    }
    // MLIR: scf.yield
}
// AST-PRETTY: )

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

// AST-PRETTY: GateDeclarationNode(name=x, params=[], qubits=[QubitNode(name=ast-gate-qubit-param-q-0, bits=0)], qcparams=[q])
// MLIR:   oq3.declare_variable @c0 : !quir.cbit<2>
// MLIR:   func @x(%arg0: !quir.qubit<1>) {
// MLIR:     return
// MLIR:   }
gate x q {}
// AST-PRETTY: GateDeclarationNode(name=cx, params=[], qubits=[QubitNode(name=ast-gate-qubit-param-q1-0, bits=0), QubitNode(name=ast-gate-qubit-param-q2-1, bits=0)], qcparams=[q1, q2])
// MLIR:   func @cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
// MLIR:     return
// MLIR:   }
gate cx q1, q2 {}

//MLIR:       %c0_i2 = arith.constant 0 : i2
//MLIR:       %0 = "oq3.cast"(%c0_i2) : (i2) -> !quir.cbit<2>
//MLIR:       oq3.variable_assign @c0 : !quir.cbit<2> = %0
bit[2] c0;
//MLIR:       %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;
//MLIR:       %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
qubit $1;

U(1.57079632679, 0.0, 3.14159265359) $0;
cx $0, $1;
c0[0] = measure $0;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=c0, bits=2), right=IntNode(signed=true, value=1, bits=32))
// AST-PRETTY: ,
// AST-PRETTY: statements=
// MLIR:       scf.while : () -> () {
// MLIR:         %5 = oq3.variable_load @c0 : !quir.cbit<2>
// MLIR:         %c1_i32 = arith.constant 1 : i32
// MLIR:         %6 = "oq3.cast"(%5) : (!quir.cbit<2>) -> i32
// MLIR:         %7 = arith.cmpi eq, %6, %c1_i32 : i32
// MLIR:         scf.condition(%7)
// MLIR:       } do {
while (c0 == 1) {
    // AST-PRETTY: GateGenericNode(name=x, params=[], qubits=[], qcparams=[$0])
    // AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=c0, bits=2)[index=0])
    // AST-PRETTY: )
    // MLIR-NO-CIRCUITS:         quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
    // MLIR-NO-CIRCUITS:         %cst = constant unit
    // MLIR-NO-CIRCUITS:         %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    // MLIR-CIRCUITS:         %5 = quir.call_circuit @circuit_1(%1) : (!quir.qubit<1>) -> i1
    // MLIR:         oq3.cbit_assign_bit @c0<2> [0] : i1 = %5
    // MLIR:         scf.yield
    // MLIR:       }
    x $0;
    c0[0] = measure $0;
}

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=c0, bits=2)[index=1])
// MLIR-CIRCUITS:      %4 = quir.call_circuit @circuit_2(%1) : (!quir.qubit<1>) -> i1
// MLIR-NO-CIRCUITS:       %4 = quir.measure(%1) : (!quir.qubit<1>) -> i1
// MLIR:       oq3.cbit_assign_bit @c0<2> [1] : i1 = %4
c0[1] = measure $0;

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

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

// MLIR: module
// MLIR-DAG: func.func @main

bit[6] r;

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset
// MLIR-DAG: oq3.declare_variable @r : !quir.cbit<6>

qubit $0;
qubit $1;
qubit $2;

// AST-PRETTY-COUNT-3: DeclarationNode(type=ASTTypeQubitContainer
// MLIR-DAG-COUNT-3: quir.declare_qubit

gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}

x $0;

r[0] = measure $0; // expected "1"
r[1] = measure $1; // expected "0"
r[2] = measure $2; // expected "1"

bit meas_and;
// MLIR-DAG: oq3.declare_variable @meas_and : !quir.cbit<1>

// MLIR: [[R:%.*]] = oq3.variable_load @r
// MLIR: [[R0:%.*]] = oq3.cbit_extractbit([[R]] : !quir.cbit<6>) [0] : i1
// MLIR: [[R:%.*]] = oq3.variable_load @r
// MLIR: [[R2:%.*]] = oq3.cbit_extractbit([[R]] : !quir.cbit<6>) [2] : i1
// MLIR-DAG: [[R0_OR_R2:%.*]] = oq3.cbit_or [[R0]], [[R2]] : i1
// MLIR: [[R:%.*]] = oq3.variable_load @r
// MLIR: [[R1:%.*]] = oq3.cbit_extractbit([[R]] : !quir.cbit<6>) [1] : i1
// MLIR-DAG: [[R0_OR_R2__AND_R1:%.*]] = oq3.cbit_and [[R0_OR_R2]], [[R1]]

// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=BinaryOpNode(type=ASTOpTypeBitAnd, left=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierRefNode(name=r[0], IdentifierNode(name=r, bits=6), index=0), right=IdentifierRefNode(name=r[2], IdentifierNode(name=r, bits=6), index=2))
// AST-PRETTY: , right=IdentifierRefNode(name=r[1], IdentifierNode(name=r, bits=6), index=1))
// AST-PRETTY: , right=IntNode(signed=true, value=1, bits=32))
if (((r[0] | r[2]) & r[1]) == 1) {
    meas_and = measure $0;
} else {
    meas_and = measure $1;
}
// MLIR: oq3.cbit_assign_bit @meas_and<1> [0] : i1 =
// on hardware, expect meas_and to become 0

if (bool(r[0] | r[1])) {
// AST-PRETTY: condition=CastNode(from=ASTTypeBinaryOp, to=ASTTypeBool, expression=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierRefNode(name=r[0], IdentifierNode(name=r, bits=6), index=0), right=IdentifierRefNode(name=r[1], IdentifierNode(name=r, bits=6), index=1))
    r[3] = measure $0;
} else {
    r[3] = measure $1;
}
// MLIR: oq3.cbit_assign_bit @r<6> [3] : i1 =

if (bool(r[0] ^ r[1]))  {
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeXor, left=IdentifierRefNode(name=r[0], IdentifierNode(name=r, bits=6), index=0), right=IdentifierRefNode(name=r[1], IdentifierNode(name=r, bits=6), index=1))
// MLIR: oq3.cbit_xor
  r[4] = measure $0;
} else {
    r[4] = measure $1;
}
// MLIR: oq3.cbit_assign_bit @r<6> [4] : i1 =
// on hardware, expect e to be 1

r[5] = 0;

r[5] = r[4] | r[3];

r[5] = !r[5];

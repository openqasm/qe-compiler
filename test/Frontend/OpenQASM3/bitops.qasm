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
// MLIR-DAG: func @main


bit a;
bit b;
bit c;

// AST-PRETTY-COUNT-2: DeclarationNode(type=ASTTypeBitset
// MLIR-DAG: quir.declare_variable @a : !quir.cbit<1>
// MLIR-DAG: quir.declare_variable @b : !quir.cbit<1>
// MLIR-DAG: quir.declare_variable @c : !quir.cbit<1>

qubit $0;
qubit $1;
qubit $2;

// AST-PRETTY-COUNT-3: DeclarationNode(type=ASTTypeQubitContainer
// MLIR-DAG-COUNT-3: quir.declare_qubit

gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}

x $0;


a = measure $0; // expected "1"
b = measure $1; // expected "0"
c = measure $2; // expected "1"

bit meas_and;
// MLIR-DAG: quir.declare_variable @meas_and : !quir.cbit<1>

// MLIR-DAG: [[A:%.*]] = quir.use_variable @a
// MLIR-DAG: [[B:%.*]] = quir.use_variable @b
// MLIR-DAG: [[C:%.*]] = quir.use_variable @c
// MLIR-DAG: [[A_OR_C:%.*]] = quir.cbit_or [[A]], [[C]] : !quir.cbit<1>
// MLIR-DAG: [[A_OR_C__AND_B:%.*]] = quir.cbit_and [[A_OR_C]], [[B]]

// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=BinaryOpNode(type=ASTOpTypeBitAnd, left=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=a, bits=1), right=IdentifierNode(name=c, bits=1))
// AST-PRETTY: , right=IdentifierNode(name=b, bits=1))
// AST-PRETTY: , right=IntNode(signed=true, value=1, bits=32))
if (((a | c) & b) == 1) {
    meas_and = measure $0;
} else {
    meas_and = measure $1;
}
// MLIR: quir.assign_cbit_bit @meas_and<1> [0] : i1 =
// on hardware, expect meas_and to become 0

bit d;

if (bool(a | b)) {
// AST-PRETTY: condition=CastNode(from=ASTTypeBinaryOp, to=ASTTypeBool, expression=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=a, bits=1), right=IdentifierNode(name=b, bits=1))
// MLIR-DAG: [[A:%.*]] = quir.use_variable @a
// MLIR-DAG: [[B:%.*]] = quir.use_variable @b
// MLIR: [[CBIT_OR_RES:%[0-9]+]] = quir.cbit_or [[A]], [[B]]
// MLIR: "quir.cast"([[CBIT_OR_RES]]) {{.*}} -> i1
    d = measure $0;
} else {
    d = measure $1;
}
// MLIR: quir.assign_cbit_bit @d<1> [0] : i1 =
// on hardware, expect d to be 1

bit e;

if (bool(a ^ b))  {
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeXor, left=IdentifierNode(name=a, bits=1), right=IdentifierNode(name=b, bits=1))
// MLIR-DAG: [[A:%.*]] = quir.use_variable @a
// MLIR-DAG: [[B:%.*]] = quir.use_variable @b
// MLIR: [[CBIT_XOR_RES:%[0-9]+]] = quir.cbit_xor [[A]], [[B]]
// MLIR: "quir.cast"([[CBIT_XOR_RES]]) {{.*}} -> i1
    e = measure $0;
} else {
    e = measure $1;
}
// MLIR: quir.assign_cbit_bit @e<1> [0] : i1 =
// on hardware, expect e to be 1

bit f = "0";

f = e | d;

// MLIR: [[F:%.*]] = quir.use_variable @f : !quir.cbit<1>
// MLIR: [[BOOL_F:%.*]] = "quir.cast"([[F]]) : (!quir.cbit<1>) -> i1
// MLIR: [[TRUE:%.*]] = arith.constant true
// MLIR: [[NOT:%.*]] = arith.cmpi ne, [[BOOL_F]], [[TRUE]] : i1
// MLIR: [[NOT_CBIT:%.*]] = "quir.cast"([[NOT]]) : (i1) -> !quir.cbit<1>
// MLIR: quir.assign_variable @f : !quir.cbit<1> = [[NOT_CBIT]]
f = !f;

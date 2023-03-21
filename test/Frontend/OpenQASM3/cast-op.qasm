OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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

qubit $0;
bit result;
// MLIR: oq3.declare_variable @result : !quir.cbit<1>
// MLIR: %[[MEASURE_RESULT:.*]] = quir.measure({{.*}} -> i1
// MLIR oq3.cbit_assign_bit @result<1> [0] : i1 =  %[[MEASURE_RESULT]]
result = measure $0;

// MLIR: %[[RESULT:.*]] = oq3.variable_load @result : !quir.cbit<1>
// MLIR: %[[COND:.*]] = "oq3.cast"(%[[RESULT]]) : (!quir.cbit<1>) -> i1
// MLIR: scf.if %[[COND]] {
if (result) {
    U(3.1415926, 0, 3.1415926) $0;
}

bit foo = "0";

// TODO: adopt fix in qss-qasm and then add a test case for bool(foo) (i.e.,
//       cast an identifier)
bool b;
// COM: AST-PRETTY: CastNode(from=ASTTypeIdentifier, to=ASTTypeBool, expression=IdentifierNode(name=foo, bits=1))
// b = bool(foo);

// MLIR-DAG: %[[FOO:.*]] = oq3.variable_load @foo : !quir.cbit<1>
// MLIR-DAG: %[[RESULT:.*]] = oq3.variable_load @result : !quir.cbit<1>
// MLIR: %[[OR:.*]] = oq3.cbit_or %[[FOO]], %[[RESULT]] : !quir.cbit<1>
// MLIR: %[[RES2:.*]] = "oq3.cast"(%[[OR]]) : (!quir.cbit<1>) -> i1
// MLIR: scf.if %[[RES2]] {
// AST-PRETTY: condition=CastNode(from=ASTTypeBinaryOp, to=ASTTypeBool, expression=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=foo, bits=1), right=IdentifierNode(name=result, bits=1))
if (bool(foo | result)) {
    foo = result;
}

// MLIR: %[[FOO:.*]] = oq3.variable_load @foo : !quir.cbit<1>
// MLIR: %[[RES:.*]] = "oq3.cast"(%[[FOO]]) : (!quir.cbit<1>) -> i1
// MLIR: scf.if %[[RES]] {
// AST-PRETTY: condition=CastNode(from=ASTTypeIdentifier, to=ASTTypeBool, expression=IdentifierNode(name=foo, bits=1)
if (bool(foo)) {
   foo = result;
}

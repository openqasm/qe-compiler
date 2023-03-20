OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
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


// This test case validates that input and output modifiers for variables are
// parsed correctly and are reflected in generated QUIR.


// Inspired by example from spec

// TODO introduce QUIR variable handling for ints
// AST-PRETTY: DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=0, bits=32), inputVariable)
input int basis;

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=flags, bits=32), inputVariable)
// MLIR-DAG: oq3.declare_variable {input} @flags : !quir.cbit<32>
input bit[32] flags;

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=result, bits=1), outputVariable)
// MLIR-DAG: oq3.declare_variable {output} @result : !quir.cbit<1>
output bit result;

// TODO
// AST-PRETTY: DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=0, bits=32), outputVariable)
output int sum2;

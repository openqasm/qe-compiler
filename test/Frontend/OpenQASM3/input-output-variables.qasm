OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This test case validates that input and output modifiers for variables are
// parsed correctly and are reflected in generated QUIR.


// Inspired by example from spec

// TODO introduce QUIR variable handling for ints
// AST-PRETTY: DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=0, bits=32), inputVariable)
input int basis;

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=flags, bits=32), inputVariable)
// MLIR-DAG: quir.declare_variable {input} @flags : !quir.cbit<32>
input bit[32] flags;

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=result, bits=1), outputVariable)
// MLIR-DAG: quir.declare_variable {output} @result : !quir.cbit<1>
output bit result;

// TODO
// AST-PRETTY: DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=0, bits=32), outputVariable)
output int sum2;

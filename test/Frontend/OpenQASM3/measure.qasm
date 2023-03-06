OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

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

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=a, bits=1))
bit a;
// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=a, bits=1))
// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0
// MLIR: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]])
// MLIR: quir.assign_cbit_bit @a<1> [0] : i1 = [[MEASURE0]]
a = measure $0;

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$1:0, bits=1)))
qubit $1;
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=b, bits=1, value=1, MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=ast-measure-result-{{.*}}, bits=1))
// AST-PRETTY: ))
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1
// MLIR: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]])
// MLIR: [[MEASURE1_CAST:%.*]] = "quir.cast"([[MEASURE1]]) : (i1) -> !quir.cbit<1>
// MLIR: quir.assign_variable @b : !quir.cbit<1> = [[MEASURE1_CAST]]
bit b = measure $1;

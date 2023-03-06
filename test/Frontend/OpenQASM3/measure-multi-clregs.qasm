OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
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

// Test that measurement results reach the correct destination, be that single
// classical bits or individual bits in classical bit registers.

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
qubit $0;
qubit $1;

// Single qubit measurement into single qubit
bit clbit;

// MLIR: [[MEASURE1:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @clbit<1> [0] : i1 = [[MEASURE1]]
clbit = measure $0;

barrier $0;

// Single qubit measurement into a multi-bit classical register
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=clreg, bits=4))
bit[4] clreg;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=1])
// MLIR: [[MEASURE2:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @clreg<4> [1] : i1 = [[MEASURE2]]
clreg[1] = measure $0;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=0])
// MLIR: [[MEASURE3:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @clreg<4> [0] : i1 = [[MEASURE3]]
clreg[0] = measure $0;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=3])
// MLIR: [[MEASURE4:%.*]] = quir.measure([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @clreg<4> [3] : i1 = [[MEASURE4]]
clreg[3] = measure $1;
//

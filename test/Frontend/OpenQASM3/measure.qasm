OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false| FileCheck %s --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s  --check-prefixes MLIR,MLIR-CIRCUITS

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

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=a, bits=1))
bit a;
// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=a, bits=1))
// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0
// MLIR-NO-CIRCUITS: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]])
// MLIR-CIRCUITS: [[MEASURE0:%.*]] = quir.call_circuit @circuit_0([[QUBIT0]])
// MLIR: oq3.cbit_assign_bit @a<1> [0] : i1 = [[MEASURE0]]
a = measure $0;

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$1:0, bits=1)))
qubit $1;
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=b, bits=1, value=1, MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=ast-measure-result-{{.*}}, bits=1))
// AST-PRETTY: ))
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1
// MLIR-NO-CIRCUITS: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]])
// MLIR-CIRCUITS: [[MEASURE1:%.*]] = quir.call_circuit @circuit_1([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR: [[MEASURE1_CAST:%.*]] = "oq3.cast"([[MEASURE1]]) : (i1) -> !quir.cbit<1>
// MLIR: oq3.variable_assign @b : !quir.cbit<1> = [[MEASURE1_CAST]]
bit b = measure $1;

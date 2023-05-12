OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
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

// Test that measurement results reach the correct destination, be that single
// classical bits or individual bits in classical bit registers.

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
qubit $0;
qubit $1;

// Single qubit measurement into single qubit
bit clbit;

// MLIR-NO-CIRCUITS: [[MEASURE1:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE1:%.*]] = quir.call_circuit @circuit_0([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @clbit<1> [0] : i1 = [[MEASURE1]]
clbit = measure $0;

barrier $0;

// Single qubit measurement into a multi-bit classical register
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=clreg, bits=4))
bit[4] clreg;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=1])
// MLIR-NO-CIRCUITS: [[MEASURE2:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE2:%.*]] = quir.call_circuit @circuit_1([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR-NO-CIRCUITS: oq3.cbit_assign_bit @clreg<4> [1] : i1 = [[MEASURE2]]
clreg[1] = measure $0;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=0])
// MLIR-NO-CIRCUITS: [[MEASURE3:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE3:%.*]] = quir.call_circuit @circuit_2([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR-NO-CIRCUITS: oq3.cbit_assign_bit @clreg<4> [0] : i1 = [[MEASURE3]]
clreg[0] = measure $0;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=3])
// MLIR-NO-CIRCUITS: [[MEASURE4:%.*]] = quir.measure([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE4:%.*]] = quir.call_circuit @circuit_3([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR-NO-CIRCUITS: oq3.cbit_assign_bit @clreg<4> [3] : i1 = [[MEASURE4]]
clreg[3] = measure $1;
//

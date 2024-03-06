OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm=false| FileCheck %s --match-full-lines --check-prefixes MLIR
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits-from-qasm | FileCheck %s --match-full-lines --check-prefixes MLIR

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

// TODO: putting resets in circuits has been disabled. The XX-tests
// are the correct tests if it is re-enabled

// XX-MLIR-CIRCUITS: quir.circuit @circuit_0(%arg0: !quir.qubit<1>) {
// XX-MLIR-CIRCUITS: quir.reset %arg0 : !quir.qubit<1>
// XX-MLIR-CIRCUITS: quir.return

// MLIR: qcs.init
// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;

// AST: <StatementList>
// AST: <ResetNode>
// AST: <GateQOpNode>
// AST: <GateOpNode>
// AST: </GateOpNode>
// AST: </GateQOpNode>
// AST: <TargetName>$0</TargetName>
// AST: </ResetNode>
// AST: </StatementList>
// AST-PRETTY: ResetNode(IdentifierNode(name=$0, bits=1))
// MLIR: quir.reset [[QUBIT0]] : !quir.qubit<1>
// XX-MLIR-NO-CIRCUITS: quir.reset [[QUBIT0]] : !quir.qubit<1>
// XX-MLIR-CIRCUITS: quir.call_circuit @circuit_0([[QUBIT0]]) : (!quir.qubit<1>) -> ()
reset $0;

// MLIR: qcs.finalize

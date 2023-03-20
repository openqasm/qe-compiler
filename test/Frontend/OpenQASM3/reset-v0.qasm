OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
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
reset $0;

// MLIR: qcs.finalize

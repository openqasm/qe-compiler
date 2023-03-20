// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
OPENQASM 3.0;

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


// AST-PRETTY: DeclarationNode(type=ASTTypeFunctionDeclaration, FunctionDeclarationNode(FunctionDefinitionNode(name=mixed_params, mangled name=_QF12mixed_paramsFrB1EFp0_II32_1iEFp1_QC1_1qEE_,
// AST-PRETTY: parameters=[DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=i, bits=32))
// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, IdentifierNode(name=q, bits=1))
// AST-PRETTY: ],
// AST-PRETTY: results=ResultNode(CBitNode(name=bitset, bits=1))
// AST-PRETTY: statements=[
// AST-PRETTY: ReturnNode(MeasureNode(qubits=[QubitContainerNode(QubitNode(name=%q:0, bits=1))], result=CBitNode(name=ast-measure-result-{{.*}}, bits=1))
// AST-PRETTY: )])

def mixed_params(int[32] i, qubit q) -> bit {
  return measure q;
}

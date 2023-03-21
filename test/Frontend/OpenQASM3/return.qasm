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

qubit $0;

// AST-PRETTY: DeclarationNode(type=ASTTypeFunctionDeclaration, FunctionDeclarationNode(FunctionDefinitionNode(name=test_measure, mangled name=_QF12test_measureFrB1EFp0_B1_9to_removeEE_,
// AST-PRETTY: ReturnNode(MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=ast-measure-result-{{.*}}-{{.*}}-{{.*}}, bits=1))
def test_measure(bit to_remove) -> bit {
  return measure $0;
}

// AST-PRETTY: ReturnNode(IdentifierNode(name=c, bits=1))])
def test_bit(bit to_remove) -> bit {
  bit c = 0;
  return c;
}

// AST-PRETTY: ReturnNode(IntNode(signed=true, value=5423, bits=32))])
def test_int(bit to_remove) -> int {
  return 5423;
}

// AST-PRETTY: ReturnNode(BinaryOpNode(type=ASTOpTypeMul, left=IntNode(signed=true, value=5423, bits=32), right=IntNode(signed=true, value=2, bits=32))
// AST-PRETTY: )])
def test_binop(bit to_remove) -> int {
  return 5423 * 2;
}

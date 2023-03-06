// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
OPENQASM 3.0;

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

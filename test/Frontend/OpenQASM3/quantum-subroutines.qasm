// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
OPENQASM 3.0;

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

// AST-PRETTY: DeclarationNode(type=ASTTypeFunctionDeclaration, FunctionDeclarationNode(FunctionDefinitionNode(name=quantum_params, mangled name=_QF14quantum_paramsFrB1EFp0_QC1_2q0EE_,
// AST-PRETTY: parameters=[DeclarationNode(type=ASTTypeQubitContainer, IdentifierNode(name=q0, bits=1))
// AST-PRETTY: ],
// AST-PRETTY: results=ResultNode(CBitNode(name=bitset, bits=1))
// AST-PRETTY: statements=[
// AST-PRETTY: ReturnNode(MeasureNode(qubits=[QubitContainerNode(QubitNode(name=%q0:0, bits=1))], result=CBitNode(name=ast-measure-result-{{.*}}, bits=1))
// AST-PRETTY: )])
def quantum_params(qubit q0) -> bit {
  return measure q0;
}

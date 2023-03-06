OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
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

// AST: <StatementList>
// AST: <Declaration>
// AST: <Identifier>result</Identifier>
// AST: <Type>ASTTypeBitset</Type>
// AST: <IsTypeDeclaration>false</IsTypeDeclaration>
// AST: <StatementNode>
// AST: <Identifier>
// AST: <Name>result</Name>
// AST: <Bits>1</Bits>
// AST: <Indexed>false</Indexed>
// AST: <NoQubit>false</NoQubit>
// AST: <RValue>false</RValue>
// AST: </Identifier>
// AST: </StatementNode>

// MLIR-DAG: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;

// MLIR-DAG: quir.declare_variable @result : !quir.cbit<1>
bit result;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=result, bits=1))
// MLIR: %[[MVAL:.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @result<1> [0] : i1 = %[[MVAL]]
result = measure $0;

// AST: <IfStatement>
// AST: <BinaryOpNode>
// AST: <Left>
// AST: <ASTExpressionNode>
// AST: <Type>ASTTypeIdentifier</Type>
// AST: <Identifier>
// AST: <Name>result</Name>
// AST: <Bits>1</Bits>
// AST: <Indexed>false</Indexed>
// AST: <Op>==</Op>
// AST: <Right>
// AST: <SignedInt>
// AST: <Identifier>ast-int-node-{{.*}}</Identifier>
// AST: <Bits>32</Bits>
// AST: <Value>1</Value>
// AST: <UGateOpNode>
// AST: <GateQOpNode>
// AST: <GateOpNode>
// AST: <UGate>
// AST: <Gate>
// AST: <Name>U</Name>
// AST: <Params>
// AST: <Angle>
// AST: <Value>3.1415926000000001</Value>
// AST: <Angle>
// AST: <Value>0.00</Value>
// AST: <Angle>
// AST: <Value>3.1415926000000001</Value>
// AST: <QubitParams>
// AST: <QubitParam>
// AST: <Name>$0</Name>
// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=result, bits=1), right=IntNode(signed=true, value=1, bits=32))
// AST-PRETTY: statements=
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=3.1415926, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.1415926, bits=64)], qubits=[], qcparams=[$0])
// AST-PRETTY: )
// MLIR: %c1_i32 = arith.constant 1 : i32
// MLIR: %{{.*}} = arith.cmpi eq, %{{.*}}, %c1_i32 : i32
// MLIR: scf.if %{{.*}} {
// MLIR: quir.builtin_U [[QUBIT0]], %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
if (result==1) {
    U(3.1415926, 0, 3.1415926) $0;
}

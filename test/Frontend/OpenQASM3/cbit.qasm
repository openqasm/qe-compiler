OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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

// MLIR-DAG: quir.declare_variable @c0 : !quir.cbit<1>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c0, bits=1))
bit c0;

// MLIR-DAG: quir.declare_variable @my_bit : !quir.cbit<1>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=my_bit, bits=1, value=1))
bit my_bit = 1;

// MLIR-DAG: quir.declare_variable @c : !quir.cbit<3>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c, bits=3))
bit[3] c;

// MLIR-DAG: quir.declare_variable @my_one_bits : !quir.cbit<2>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=my_one_bits, bits=2, value=11))
bit[2] my_one_bits = "11";

// MLIR-DAG: quir.declare_variable @my_bitstring : !quir.cbit<10>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=my_bitstring, bits=10, value=1101001010))
bit[10] my_bitstring = "1101001010";

// MLIR-DAG: quir.declare_variable @result : !quir.cbit<20>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=result, bits=20))
bit[20] result;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=result, bits=20)[index=6])
// MLIR: %[[QUBIT:.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: %[[MEASUREMENT:.*]] = quir.measure(%[[QUBIT]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @result<20> [6] : i1 = %[[MEASUREMENT]]
qubit $0;
result[6] = measure $0;

// AST-PRETTY: condition=IdentifierNode(name=my_one_bits, bits=2),
// MLIR: %[[MY_ONE_BITS:.*]] = quir.use_variable @my_one_bits : !quir.cbit<2>
// MLIR: %{{.*}} = "quir.cast"(%[[MY_ONE_BITS]]) : (!quir.cbit<2>) -> i1
if (my_one_bits) {
    U(3.1415926, 0, 3.1415926) $0;
}

// AST-PRETTY: condition=IdentifierRefNode(name=result[6], IdentifierNode(name=result, bits=20), index=6),
// MLIR: %[[RESULT:.*]] = quir.use_variable @result : !quir.cbit<20>
// MLIR: %[[LVAL:.*]] = quir.cbit_extractbit(%[[RESULT]] : !quir.cbit<20>) [6] : i1
// MLIR: scf.if %[[LVAL]] {
if (result[6]) {
    U(3.1415926, 0, 3.1415926) $0;
}

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

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=bs, bits=32, value=00000000000000000000000000000001))
bit[32] bs = 1;

// AST-PRETTY: ForStatementNode(start=0, end=4,
// MLIR: scf.for %arg1 = %c0_0 to %c5 step %c1_1 {
for i in [0 : 4] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = quir.use_variable @bs : !quir.cbit<32>
    // MLIR: %2 = "quir.cast"(%arg1) : (index) -> !quir.cbit<32>
    // MLIR: %3 = quir.cbit_or %1, %2 : !quir.cbit<32>
    // MLIR: quir.assign_variable @bs : !quir.cbit<32> = %3
    bs = bs | i;
}

// AST-PRETTY: ForStatementNode(start=0, end=3,
// MLIR: scf.for %arg1 = %c0_2 to %c4 step %c1_3 {
for i in [0 : 3] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeBitAnd, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = quir.use_variable @bs : !quir.cbit<32>
    // MLIR: %2 = "quir.cast"(%arg1) : (index) -> !quir.cbit<32>
    // MLIR: %3 = quir.cbit_and %1, %2 : !quir.cbit<32>
    // MLIR: quir.assign_variable @bs : !quir.cbit<32> = %3
    bs = bs & i;
}

// AST-PRETTY: ForStatementNode(start=0, end=5,
// MLIR: scf.for %arg1 = %c0_4 to %c6 step %c1_5 {
for i in [0 : 5] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeXor, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = quir.use_variable @bs : !quir.cbit<32>
    // MLIR: %2 = "quir.cast"(%arg1) : (index) -> !quir.cbit<32>
    // MLIR: %3 = quir.cbit_xor %1, %2 : !quir.cbit<32>
    // MLIR: quir.assign_variable @bs : !quir.cbit<32> = %3
    bs = bs ^ i;
}

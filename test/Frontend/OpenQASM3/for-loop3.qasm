OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=bs, bits=32, value=00000000000000000000000000000001))
bit[32] bs = 1;

// AST-PRETTY: ForStatementNode(start=0, end=4,
// MLIR: scf.for %arg1 = %c0_0 to %c5 step %c1_1 {
for i in [0 : 4] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = oq3.variable_load @bs : !quir.cbit<32>
    // MLIR: %2 = "oq3.cast"(%arg1) : (index) -> !quir.cbit<32>
    // MLIR: %3 = oq3.cbit_or %1, %2 : !quir.cbit<32>
    // MLIR: oq3.variable_assign @bs : !quir.cbit<32> = %3
    bs = bs | i;
}

// AST-PRETTY: ForStatementNode(start=0, end=3,
// MLIR: scf.for %arg1 = %c0_2 to %c4 step %c1_3 {
for i in [0 : 3] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeBitAnd, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = oq3.variable_load @bs : !quir.cbit<32>
    // MLIR: %2 = "oq3.cast"(%arg1) : (index) -> !quir.cbit<32>
    // MLIR: %3 = oq3.cbit_and %1, %2 : !quir.cbit<32>
    // MLIR: oq3.variable_assign @bs : !quir.cbit<32> = %3
    bs = bs & i;
}

// AST-PRETTY: ForStatementNode(start=0, end=5,
// MLIR: scf.for %arg1 = %c0_4 to %c6 step %c1_5 {
for i in [0 : 5] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeXor, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = oq3.variable_load @bs : !quir.cbit<32>
    // MLIR: %2 = "oq3.cast"(%arg1) : (index) -> !quir.cbit<32>
    // MLIR: %3 = oq3.cbit_xor %1, %2 : !quir.cbit<32>
    // MLIR: oq3.variable_assign @bs : !quir.cbit<32> = %3
    bs = bs ^ i;
}

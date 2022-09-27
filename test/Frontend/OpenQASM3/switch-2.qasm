OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY

int i = 15;

int j = 1;
int k = 2;

bit c1;

qubit[8] $0;
qubit $1;

// AST-PRETTY: SwitchStatementNode(SwitchQuantity(name=i, type=ASTTypeIdentifier),
switch (i) {
    // AST-PRETTY: statements=[
    // AST-PRETTY: CaseStatementNode(case=1, BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=j, bits=32), right=IdentifierNode(name=k, bits=32))
    case 1: {
        j = k;
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=2, DeclarationNode(type=ASTTypeMPDecimal, IdentifierNode(name=d, bits=64))
    case 2: {
        float[64] d = j / k;
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=3, ),
    case 3: {
    }
    break;
    // AST-PRETTY: ],
    // AST-PRETTY: default statement=[
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=j, bits=32), right=BinaryOpNode(type=ASTOpTypeAdd, left=BinaryOpNode(type=ASTOpTypeAdd, left=IdentifierNode(name=i, bits=32), right=IdentifierNode(name=j, bits=32))
    // AST-PRETTY: , right=IdentifierNode(name=k, bits=32))
    // AST-PRETTY: )
    // AST-PRETTY: ])
    default: {
        j = i + j + k;
    }
    break;
}

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY

int i = 15;

// AST-PRETTY: SwitchStatementNode(SwitchQuantity(name=i, type=ASTTypeIdentifier),
switch (i) {
    // AST-PRETTY: statements=[
    // AST-PRETTY: CaseStatementNode(case=1, ),
    case 1: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=2, ),
    case 2: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=3, ),
    case 3: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=5, ),
    case 5: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=12, ),
    case 12: {
    }
    break;
    // AST-PRETTY: ],
    // AST-PRETTY: default statement=[
    // AST-PRETTY: ])
    default: {
    }
    break;
}

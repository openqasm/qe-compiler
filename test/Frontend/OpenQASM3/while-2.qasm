OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

qubit $0;
int n = 5;
int i = 5;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeGT, left=IdentifierNode(name=n, bits=32), right=IntNode(signed=true, value=0, bits=32))
// error: illegal hardware instruction
while (n > 0) {
    h $0;
    // while (i > 0) {
    //     h $0;
    //     // infinite loop with error: Binary operation ASTOpTypeSub not supported yet.
    //     // i = i - 1;
    // }
    // error: Binary operation ASTOpTypeSub not supported yet.
    // n = n - 1;
}
// AST-PRETTY: )

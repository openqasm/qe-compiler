OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

qubit $0;
int n = 5;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeGT, left=IdentifierNode(name=n, bits=32), right=IntNode(signed=true, value=0, bits=32))
while (n > 0) {
    // AST-PRETTY: statements=
    // AST-PRETTY: HGateOpNode(params=[], qubits=[], qcparams=[$0])
    h $0;
}
// AST-PRETTY: )

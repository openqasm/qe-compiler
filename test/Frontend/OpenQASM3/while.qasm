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
    // AST-PRETTY: HGateOpNode(params=[], qubits=[], qcparams=[$0],
    // AST-PRETTY: ops=[
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
    h $0;
}
// AST-PRETTY: )

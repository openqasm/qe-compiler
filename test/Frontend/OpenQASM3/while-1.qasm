OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

qubit $0;
int n = 1;

bit is_excited;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=n, bits=32), right=IntNode(signed=true, value=0, bits=32))
// MLIR: scf.while : () -> () {
// MLIR:     %2 = quir.use_variable @n : i32
// MLIR:     %c0_i32_0 = arith.constant 0 : i32
// MLIR:     %3 = arith.cmpi ne, %2, %c0_i32_0 : i32
// MLIR:     scf.condition(%3)
// MLIR: } do {
while (n != 0) {
    // AST-PRETTY: statements=
    // AST-PRETTY: HGateOpNode(params=[], qubits=[], qcparams=[$0])
    // MLIR: quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
    // MLIR: %cst = constant unit
    h $0;
    // MLIR: %2 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // MLIR: quir.assign_cbit_bit @is_excited<1> [0] : i1 = %2
    // MLIR: %3 = quir.use_variable @is_excited : !quir.cbit<1>
    // MLIR: %4 = "quir.cast"(%3) : (!quir.cbit<1>) -> i1
    is_excited = measure $0;
    // MLIR: scf.if %4 {
    if (is_excited) {
        // MLIR:     quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
        // MLIR:     %cst_0 = constant unit
        // MLIR: }
        // MLIR: scf.yield
        h $0;
    }
    // error: Binary operation ASTOpTypeSub not supported yet.
    // n = n - 1;
    // MLIR: %c0_i32_0 = arith.constant 0 : i32
    // MLIR: quir.assign_variable @n : i32 = %c0_i32_0
    n = 0;
    // MLIR: scf.yield
}
// AST-PRETTY: )

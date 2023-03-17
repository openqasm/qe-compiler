OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

qubit $0;
int n = 2;
bit is_excited;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=n, bits=32), right=IntNode(signed=true, value=0, bits=32))
// MLIR: scf.while : () -> () {
// MLIR:     %2 = oq3.variable_load @n : i32
// MLIR:     %c0_i32_0 = arith.constant 0 : i32
// MLIR:     %3 = arith.cmpi ne, %2, %c0_i32_0 : i32
// MLIR:     scf.condition(%3)
// MLIR: } do {
while (n != 0) {
    // MLIR: %2 = oq3.variable_load @n : i32
    // MLIR: %c2_i32_0 = arith.constant 2 : i32
    // MLIR: %3 = arith.cmpi eq, %2, %c2_i32_0 : i32
    // MLIR: scf.if %3 {
    if (n == 2) {
        // MLIR: quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
        // MLIR: %cst = constant unit
        // MLIR: %c1_i32 = arith.constant 1 : i32
        // MLIR: oq3.variable_assign @n : i32 = %c1_i32
        h $0;
        n = 1;
    // MLIR: } else {
    } else {
        // MLIR: quir.call_gate @h(%0) : (!quir.qubit<1>) -> ()
        // MLIR: %cst = constant unit
        // MLIR: %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
        // MLIR: oq3.cbit_assign_bit @is_excited<1> [0] : i1 = %4
        // MLIR: %c0_i32_1 = arith.constant 0 : i32
        // MLIR: oq3.variable_assign @n : i32 = %c0_i32_1
        h $0;
        is_excited = measure $0;
        n = 0;
    }
    // MLIR: scf.yield
}
// AST-PRETTY: )

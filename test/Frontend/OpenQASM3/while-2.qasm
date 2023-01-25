OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

qubit $0;
int n = 2;
bit is_excited;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=n, bits=32), right=IntNode(signed=true, value=0, bits=32))
// MLIR: scf.while : () -> () {
// MLIR:     %2 = quir.use_variable @n : i32
// MLIR:     %c0_i32_0 = arith.constant 0 : i32
// MLIR:     %3 = arith.cmpi ne, %2, %c0_i32_0 : i32
// MLIR:     scf.condition(%3)
// MLIR: } do {
while (n != 0) {
    // MLIR: %2 = quir.use_variable @n : i32
    // MLIR: %c2_i32_0 = arith.constant 2 : i32
    // MLIR: %3 = arith.cmpi eq, %2, %c2_i32_0 : i32
    // MLIR: scf.if %3 {
    if (n == 2) {
        // MLIR: %angle = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
        // MLIR: %angle_1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
        // MLIR: %angle_2 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
        // MLIR: quir.builtin_U %0, %angle, %angle_1, %angle_2 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64> 
        U(1.57079632679, 0.0, 3.14159265359) $0;
        // MLIR: %c1_i32 = arith.constant 1 : i32
        // MLIR: quir.assign_variable @n : i32 = %c1_i32
        n = 1;
    // MLIR: } else {
    } else {
        // MLIR: %angle = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
        // MLIR: %angle_1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
        // MLIR: %angle_2 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
        // MLIR: quir.builtin_U %0, %angle, %angle_1, %angle_2 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
        U(1.57079632679, 0.0, 3.14159265359) $0;
        // MLIR: %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
        // MLIR: quir.assign_cbit_bit @is_excited<1> [0] : i1 = %4
        is_excited = measure $0;
        // MLIR: %c0_i32_1 = arith.constant 0 : i32
        // MLIR: quir.assign_variable @n : i32 = %c0_i32_1
        n = 0;
    }
    // MLIR: scf.yield
}
// AST-PRETTY: )

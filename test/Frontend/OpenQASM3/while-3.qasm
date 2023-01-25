OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

qubit $0;
int i = 1;
int j = 0;
bit is_excited;

// AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=i, bits=32), right=IntNode(signed=true, value=0, bits=32))
// AST-PRETTY: ,
// AST-PRETTY: statements=
// MLIR: scf.while : () -> () {
// MLIR:     %2 = quir.use_variable @i : i32
// MLIR:     %c0_i32_1 = arith.constant 0 : i32
// MLIR:     %3 = arith.cmpi ne, %2, %c0_i32_1 : i32
// MLIR:     scf.condition(%3)
// MLIR: } do {
while (i != 0) {
    // AST-PRETTY: WhileStatement(condition=BinaryOpNode(type=ASTOpTypeCompNeq, left=IdentifierNode(name=j, bits=32), right=IntNode(signed=true, value=0, bits=32))
    // AST-PRETTY: ,
    // MLIR:     scf.while : () -> () {
    // MLIR:         %3 = quir.use_variable @j : i32
    // MLIR:         %c0_i32_4 = arith.constant 0 : i32
    // MLIR:         %4 = arith.cmpi ne, %3, %c0_i32_4 : i32
    // MLIR:         scf.condition(%4)
    // MLIR:     } do {
    while (j != 0) {
        // MLIR: %angle_4 = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
        // MLIR: %angle_5 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
        // MLIR: %angle_6 = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
        // MLIR: quir.builtin_U %0, %angle_4, %angle_5, %angle_6 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
        U(1.57079632679, 0.0, 3.14159265359) $0;
        // MLIR: %c0_i32_7 = arith.constant 0 : i32
        // MLIR: quir.assign_variable @j : i32 = %c0_i32_7
        j = 0;
        // MLIR: scf.yield
    }
    // MLIR:     %angle = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    // MLIR:     %angle_1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    // MLIR:     %angle_2 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    // MLIR:     quir.builtin_U %0, %angle, %angle_1, %angle_2 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    U (0, 0, 0) $0;
    // MLIR:     %2 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    // MLIR:     quir.assign_cbit_bit @is_excited<1> [0] : i1 = %2
    is_excited = measure $0;
    // MLIR:     %c0_i32_3 = arith.constant 0 : i32
    // MLIR:     quir.assign_variable @i : i32 = %c0_i32_3
    // MLIR:     scf.yield
    // MLIR: }
    i = 0;
}

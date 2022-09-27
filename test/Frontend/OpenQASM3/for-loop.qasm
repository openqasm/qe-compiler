OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;

// For loop.
// MLIR: %c0_0 = arith.constant 0 : index
// MLIR: %c4 = arith.constant 4 : index
// MLIR: %c1_1 = arith.constant 1 : index
// MLIR: scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
// AST-PRETTY: ForStatementNode(start=0, end=4,
for i in [0 : 4] {
    // MLIR: {{.*}} = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    // MLIR: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    // MLIR: {{.*}} = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    // MLIR: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    // AST-PRETTY: statements=
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[$0])
    // AST-PRETTY: )
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

// MLIR: %c0_2 = arith.constant 0 : index
// MLIR: %c4_3 = arith.constant 4 : index
// MLIR: %c1_4 = arith.constant 1 : index
// MLIR: scf.for %arg1 = %c0_2 to %c4_3 step %c1_4 {
// AST-PRETTY: ForStatementNode(start=0, stepping=1, end=4,
for i in [0 : 1 : 4] {
    // MLIR: {{.*}} = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    // MLIR: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    // MLIR: {{.*}} = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    // MLIR: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

// MLIR: %c-1 = arith.constant -1 : index
// MLIR: %c1000_5 = arith.constant 1000 : index
// MLIR: %c10 = arith.constant 10 : index
// MLIR: scf.for %arg1 = %c-1 to %c1000_5 step %c10 {
// AST-PRETTY: ForStatementNode(start=-1, stepping=10, end=1000,
for i in [-1 : 10 : 1000] {
    // MLIR: {{.*}} = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
    // MLIR: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    // MLIR: {{.*}} = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    // MLIR: quir.builtin_U {{.*}}, {{.*}}, {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

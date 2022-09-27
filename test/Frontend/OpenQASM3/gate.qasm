OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR


// MLIR: func @h(%arg0: !quir.qubit<1>) {
// MLIR: %angle = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
// MLIR: quir.builtin_U %arg0, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

// MLIR: func @main() -> i32 {

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;

// MLIR: quir.builtin_U [[QUBIT0]], %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
U(3.14, 0, 3.14) $0;
// MLIR: quir.call_gate @h([[QUBIT0]]) : (!quir.qubit<1>) -> ()
h $0;

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

qubit $0;
qubit $1;
// MLIR: quir.barrier %{{.*}} : (!quir.qubit<1>) -> ()
barrier $0;
// MLIR: quir.barrier %{{.*}}, %{{.*}} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
barrier $0, $1;

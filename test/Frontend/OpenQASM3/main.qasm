OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// MLIR: module {
// MLIR: func @main() -> i32 {

qubit $0;

// MLIR: return %c0_i32 : i32

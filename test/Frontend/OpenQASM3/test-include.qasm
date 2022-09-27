OPENQASM 3.0;
// RUN: qss-compiler -I=%S %s --emit=mlir | FileCheck %s

// CHECK: func @rz
include "test-include.inc";
qubit $0;
rz(0) $0;

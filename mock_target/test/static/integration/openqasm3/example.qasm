OPENQASM 3.0;
// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s
// CHECK: Manifest

qubit $0;
bit c0;
U(1.57079632679, 0.0, 3.14159265359) $0;
measure $0 -> c0;

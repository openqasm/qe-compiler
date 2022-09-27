OPENQASM 3.0;
// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s

// CHECK: Manifest
qubit $0;
qubit $1;

gate cx control, target { }

bit c0;
bit c1;

U(1.57079632679, 0.0, 3.14159265359) $0;
cx $0, $1;
measure $0 -> c0;
measure $1 -> c1;

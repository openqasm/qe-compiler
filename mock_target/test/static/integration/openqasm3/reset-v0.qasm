OPENQASM 3.0;
// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s

// CHECK: Manifest
qubit $0;
reset $0;

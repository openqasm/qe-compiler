OPENQASM 3.0;
// Ensure piping of input also works with diagnostics.
// TODO: Once https://github.com/openqasm/qe-qasm/issues/35
// is fixed this test will error and the workaround
// in OpenQASM3Frontend.cpp for line number offsets
// will need to be removed for the test to pass.
// RUN: cat %s | ( qss-compiler -X=qasm --emit=mlir || true ) 2>&1 | FileCheck %s --dump-input=fail
// RUN: ( qss-compiler -X=qasm --emit=mlir %s || true ) 2>&1 | FileCheck %s --dump-input=fail

int a;
int b;

a &&& b;
// CHECK: 13:5: error: syntax error, unexpected '&'
// CHECK: a &&& b;
// CHECK:     ^

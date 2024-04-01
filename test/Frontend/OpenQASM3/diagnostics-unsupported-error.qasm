OPENQASM 3.0;
// Ensure piping of input also works with diagnostics.
// TODO: Once https://github.com/openqasm/qe-qasm/issues/35
// is fixed this test will error and the workaround
// in QUIRGenQASM3Visitor.cpp
// for line number offsets
// will need to be removed for the test to pass.
// RUN: cat %s | ( qss-compiler -X=qasm --emit=mlir || true ) 2>&1 | FileCheck %s --dump-input=fail
// RUN: ( qss-compiler -X=qasm --emit=mlir %s  || true ) 2>&1 | FileCheck %s --dump-input=fail

int a;
float b;
int c;
c =  float(a) + b;

// CHECK: error: Unsupported cast destination type ASTTypeFloat
// CHECK: c =  float(a) + b;
// CHECK:      ^
// CHECK: error: Addition is not supported on value of type: 'none'
// CHECK: c =  float(a) + b;
// CHECK:      ^

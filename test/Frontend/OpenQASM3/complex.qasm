OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// MLIR: quir.declare_variable @a : complex<f32>
// MLIR: quir.declare_variable @my_complex : complex<f80>
// MLIR: quir.declare_variable @b : complex<f64>

// MLIR: %cst = arith.constant 3.000000e+00 : f32
// MLIR: %cst_0 = arith.constant 7.350000e+00 : f32
// MLIR: %{{.*}} = complex.create %cst, %cst_0 : complex<f32>
// MLIR: quir.assign_variable @a : complex<f32> = %{{.*}}
// AST-PRETTY: DeclarationNode(type=ASTTypeMPComplex, MPComplexNode(name=a, value=3.0000000000 + 7.3499999996 im, bits=32))
complex[32] a = 3 + 7.35 im;

// MLIR: %cst_1 = arith.constant 3.000000e+03 : f80
// MLIR: %cst_2 = arith.constant 7.35232200000000002404 : f80
// MLIR: %{{.*}} = complex.create %cst_1, %cst_2 : complex<f80>
// MLIR: quir.assign_variable @my_complex : complex<f80> = %{{.*}}
// AST-PRETTY: DeclarationNode(type=ASTTypeMPComplex, MPComplexNode(name=my_complex, value=3.00000000000000000000e+3 + 7.35232200000000002404 im, bits=65))
complex[65] my_complex = 3000 + 7.352322 im;

// MLIR: %{{.*}} = quir.use_variable @a : complex<f32>
// MLIR: [[CAST_1:%.*]] = "quir.cast"(%{{.*}}) : (complex<f32>) -> complex<f80>
// MLIR: quir.assign_variable @my_complex : complex<f80> = [[CAST_1]]
my_complex = a;

// MLIR: %cst_3 = arith.constant 0.000000e+00 : f64
// MLIR: %cst_4 = arith.constant 0.000000e+00 : f64
// MLIR: %{{.*}} = complex.create %cst_3, %cst_4 : complex<f64>
// MLIR: quir.assign_variable @b : complex<f64> = %{{.*}}
// AST-PRETTY: DeclarationNode(type=ASTTypeMPComplex, MPComplexNode(name=b, bits=64))
complex[64] b;

// The following raises error
// loc("../qss-compiler/test/Visitor/complex.qasm":36:24): error: Cannot support float with 300 bits
// Error: Failed to emit QUIR
// complex[300] c;

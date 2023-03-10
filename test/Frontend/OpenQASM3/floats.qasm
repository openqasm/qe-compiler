OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// MLIR: oq3.declare_variable @x : f32
// MLIR: oq3.declare_variable @y : f32
// MLIR: oq3.declare_variable @my_new_float : f64
// MLIR: oq3.declare_variable @a : f32
// MLIR: oq3.declare_variable @b : f32
// MLIR: oq3.declare_variable @c : f80
// MLIR: oq3.declare_variable @d : f128
// MLIR: oq3.declare_variable @e : f32
// MLIR: oq3.declare_variable @f : f32

// MLIR: %cst = arith.constant 5.500000e+00 : f32
// MLIR: oq3.variable_assign @x : f32 = %cst
// MLIR: %cst_0 = arith.constant 5.500000e+00 : f32
// MLIR: oq3.variable_assign @y : f32 = %cst_0
// AST-PRETTY: DeclarationNode(type=ASTTypeMPDecimal, MPDecimalNode(name=x, value=5.5, bits=32))
// AST-PRETTY: DeclarationNode(type=ASTTypeMPDecimal, MPDecimalNode(name=y, value=5.5, bits=32))
float[32] x = 5.5;
float[32] y = 5.5;

// MLIR: [[VAR_X_0:%.*]] = oq3.use_variable @x : f32
// MLIR: oq3.variable_assign @y : f32 = [[VAR_X_0]]
y = x;

// MLIR: %cst_1 = arith.constant 2020.2021 : f64
// MLIR: oq3.variable_assign @my_new_float : f64 = %cst_1
// AST-PRETTY: DeclarationNode(type=ASTTypeMPDecimal, MPDecimalNode(name=my_new_float, value=2020.20209999999999995, bits=64))
float[64] my_new_float = 2020.2021;

// MLIR: %cst_2 = arith.constant 0.000000e+00 : f32
// MLIR: oq3.variable_assign @a : f32 = %cst_2
// MLIR: %cst_3 = arith.constant 0.000000e+00 : f32
// MLIR: oq3.variable_assign @b : f32 = %cst_3
// MLIR: %cst_4 = arith.constant 0.000000e+00 : f80
// MLIR: oq3.variable_assign @c : f80 = %cst_4
// MLIR: %cst_5 = arith.constant 0.000000e+00 : f128
// MLIR: oq3.variable_assign @d : f128 = %cst_5
// AST-PRETTY: DeclarationNode(type=ASTTypeMPDecimal, MPDecimalNode(name=a, bits=32))
float[3] a;
float[9] b;
float[80] c;
float[81] d;

// The remainder of this test case will be re-activated as part of IBM-Q-Software/QSS-Compiler#220
// COM: MLIR: %{{.*}} = cmpi eq, %cst, %cst_0 : f32
// COM: AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=x, bits=32), right=IdentifierNode(name=y, bits=32)),
qubit $0;
// if (x == y) {
//   U(0, 0, 0) $0;
// }

// MLIR: %cst_6 = arith.constant 0.000000e+00 : f32
// MLIR: oq3.variable_assign @e : f32 = %cst_6
// AST-PRETTY: DeclarationNode(type=ASTTypeFloat, FloatNode(name=e, value=0, bits=32))
float e;

// MLIR: %cst_7 = arith.constant 1.010000e+01 : f32
// MILR: oq3.variable_assign @f : f32 = %cst_7
// AST-PRETTY: DeclarationNode(type=ASTTypeFloat, FloatNode(name=f, value=10.1, bits=32))
float f = 10.1;

// The following raises error
// loc("../qss-compiler/test/Visitor/floats.qasm":65:13): error: Cannot support float with 300 bits
// Error: Failed to emit QUIR
// float[300] g;

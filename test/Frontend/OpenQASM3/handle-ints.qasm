OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR
// MLIR: oq3.declare_variable @x : i32
// MLIR: oq3.declare_variable @y : i32
// MLIR: oq3.declare_variable @a : i3
// MLIR: oq3.declare_variable @b : i32

// MLIR: %c100_i32 = arith.constant 100 : i32
// MLIR: oq3.variable_assign @x : i32 = %c100_i32
// MLIR: %c100_i32_0 = arith.constant 100 : i32
// MLIR: oq3.variable_assign @y : i32 = %c100_i32_0
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=x, value=100, bits=32, signed=1))
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=y, value=100, bits=32, signed=1))
int[32] x = 100;
int[32] y = 100;

// MLIR: %c1_i3 = arith.constant 1 : i3
// MLIR: oq3.variable_assign @a : i3 = %c1_i3
// MLIR: %c4_i32 = arith.constant 4 : i32
// MLIR: oq3.variable_assign @b : i32 = %c4_i32
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=a, value=1, bits=3, signed=1))
int[3] a = 1;
int b = 4;

qubit $0;
qubit $1;

// MLIR: [[USE_X:%.*]] = oq3.use_variable @x : i32
// MLIR: [[USE_Y:%.*]] = oq3.use_variable @y : i32
// MLIR: %{{.*}} = arith.cmpi eq, [[USE_X]], [[USE_Y]] : i32
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=x, bits=32), right=IdentifierNode(name=y, bits=32))
if (x == y) {
// MLIR: %c2_i32_1 = arith.constant 2 : i32
// MLIR: [[CAST_A:%.*]] = "oq3.cast"(%c2_i32_1) : (i32) -> i3
// MLIR: oq3.variable_assign @a : i3 = [[CAST_A]]
// MLIR: %c3_i32_2 = arith.constant 3 : i32
// MLIR: oq3.variable_assign @b : i32 = %c3_i32_2
  a = 2;
  b = 3;
  U(0, 0, 0) $0;
}

// MLIR: [[USE_A:%.*]] = oq3.use_variable @a : i3
// MLIR: %c2_i32 = arith.constant 2 : i32
// MLIR: [[CAST_A:%.*]] = "oq3.cast"([[USE_A]]) : (i3) -> i32
// MLIR: %{{.*}} = arith.cmpi eq, [[CAST_A]], %c2_i32 : i32
if (a == 2) {
  U(0, 0, 0) $1;
}

// MLIR: [[USE_B:%.*]] = oq3.use_variable @b : i32
// MLIR: %c3_i32 = arith.constant 3 : i32
// MLIR:  %{{.*}} = arith.cmpi eq, [[USE_B]], %c3_i32 : i32
if (b == 3) {
  U(0, 0, 0) $0;
}

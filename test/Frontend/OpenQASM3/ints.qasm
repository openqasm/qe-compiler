OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// MLIR:   oq3.declare_variable @x : i32
// MLIR:   oq3.declare_variable @y : i32
// MLIR:   oq3.declare_variable @my_new_int : i64
// MLIR:   oq3.declare_variable @a : i3
// MLIR:   oq3.declare_variable @b : i9
// MLIR:   oq3.declare_variable @c : i80
// MLIR:   oq3.declare_variable @d : i81
// MLIR:   oq3.declare_variable @ux : i32
// MLIR:   oq3.declare_variable @uy : i32
// MLIR:   oq3.declare_variable @my_uint : i64
// MLIR:   oq3.declare_variable @ua : i3
// MLIR:   oq3.declare_variable @ub : i9
// MLIR:   oq3.declare_variable @uc : i80
// MLIR:   oq3.declare_variable @ud : i81

// MLIR: %c55_i32 = arith.constant 55 : i32
// MLIR: oq3.variable_assign @x : i32 = %c55_i32
// MLIR: %c55_i32_0 = arith.constant 55 : i32
// MLIR: oq3.variable_assign @y : i32 = %c55_i32_0
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=x, value=55, bits=32, signed=1))
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=y, value=55, bits=32, signed=1))
int[32] x = 55;
int[32] y = 55;

// MLIR: %c2020_i64 = arith.constant 2020 : i64
// MLIR: oq3.variable_assign @my_new_int : i64 = %c2020_i64
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=my_new_int, value=2020, bits=64, signed=1))
int[64] my_new_int = 2020;

// MLIR: %c0_i3 = arith.constant 0 : i3
// MLIR: oq3.variable_assign @a : i3 = %c0_i3
// MLIR: %c0_i9 = arith.constant 0 : i9
// MLIR: oq3.variable_assign @b : i9 = %c0_i9
// MLIR: %c0_i80 = arith.constant 0 : i80
// MLIR: oq3.variable_assign @c : i80 = %c0_i80
// MLIR: %c0_i81 = arith.constant 0 : i81
// MLIR: oq3.variable_assign @d : i81 = %c0_i81
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=a, value=0, bits=3, signed=1))
int[3] a;
int[9] b;
int[80] c;
int[81] d;

// MLIR: [[USE_X:%.*]] = oq3.use_variable @x : i32
// MLIR: [[USE_Y:%.*]] = oq3.use_variable @y : i32
// MLIR: %{{.*}} = arith.cmpi eq, [[USE_X]], [[USE_Y]] : i32
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=x, bits=32), right=IdentifierNode(name=y, bits=32))
qubit $0;
if (x == y) {
  U(0, 0, 0) $0;
}

// MLIR: %c56_i32 = arith.constant 56 : i32
// MLIR: oq3.variable_assign @ux : i32 = %c56_i32
// MLIR: %c57_i32 = arith.constant 57 : i32
// MLIR: oq3.variable_assign @uy : i32 = %c57_i32
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=ux, value=56, bits=32, signed=0))
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=uy, value=57, bits=32, signed=0))
uint[32] ux = 56;
uint[32] uy = 57;

// MLIR: %c2554_i64 = arith.constant 2554 : i64
// MLIR: oq3.variable_assign @my_uint : i64 = %c2554_i64
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=my_uint, value=2554, bits=64, signed=0))
uint[64] my_uint = 2554;

// MLIR: %c0_i3_1 = arith.constant 0 : i3
// MLIR: oq3.variable_assign @ua : i3 = %c0_i3_1
// MLIR: %c0_i9_2 = arith.constant 0 : i9
// MLIR: oq3.variable_assign @ub : i9 = %c0_i9_2
// MLIR: %c0_i80_3 = arith.constant 0 : i80
// MLIR: oq3.variable_assign @uc : i80 = %c0_i80_3
// MLIR: %c0_i81_4 = arith.constant 0 : i81
// MLIR: oq3.variable_assign @ud : i81 = %c0_i81_4
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, MPIntegerNode(name=ua, value=0, bits=3, signed=0))
uint[3] ua;
uint[9] ub;
uint[80] uc;
uint[81] ud;

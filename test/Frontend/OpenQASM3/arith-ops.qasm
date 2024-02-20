OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR



int i0 = 1;
int i1 = 2;
int i2;

float f0 = 1.0;
float f1 = 2.0;
float f2;

// Addition

i2 = i0 + i1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=i2, bits=32), right=BinaryOpNode(type=ASTOpTypeAdd, left=IdentifierNode(name=i0, bits=32), right=IdentifierNode(name=i1, bits=32))
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[i2:.*]] = arith.addi %[[i0]], %[[i1]] : i32

f2 = f0 + f1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=BinaryOpNode(type=ASTOpTypeAdd, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=f1, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[f2:.*]] = arith.addf %[[f0]], %[[f1]] : f32


// Subtraction

i2 = i0 - i1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=i2, bits=32), right=BinaryOpNode(type=ASTOpTypeSub, left=IdentifierNode(name=i0, bits=32), right=IdentifierNode(name=i1, bits=32))
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[i2:.*]] = arith.subi %[[i0]], %[[i1]] : i32

f2 = f0 - f1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=BinaryOpNode(type=ASTOpTypeSub, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=f1, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[f2:.*]] = arith.subf %[[f0]], %[[f1]] : f32


// Multiplication

i2 = i0 * i1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=i2, bits=32), right=BinaryOpNode(type=ASTOpTypeMul, left=IdentifierNode(name=i0, bits=32), right=IdentifierNode(name=i1, bits=32))
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[i2:.*]] = arith.muli %[[i0]], %[[i1]] : i32

f2 = f0 * f1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=BinaryOpNode(type=ASTOpTypeMul, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=f1, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[f2:.*]] = arith.mulf %[[f0]], %[[f1]] : f32


// Division

i2 = i0 / i1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=i2, bits=32), right=BinaryOpNode(type=ASTOpTypeDiv, left=IdentifierNode(name=i0, bits=32), right=IdentifierNode(name=i1, bits=32))
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[i2:.*]] = arith.divsi %[[i0]], %[[i1]] : i32

f2 = f0 / f1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=BinaryOpNode(type=ASTOpTypeDiv, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=f1, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[f2:.*]] = arith.divf %[[f0]], %[[f1]] : f32


// Modulo

i2 = i0 % i1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=i2, bits=32), right=BinaryOpNode(type=ASTOpTypeMod, left=IdentifierNode(name=i0, bits=32), right=IdentifierNode(name=i1, bits=32))
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[i2:.*]] = arith.remsi %[[i0]], %[[i1]] : i32

f2 = f0 % f1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=BinaryOpNode(type=ASTOpTypeMod, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=f1, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[f2:.*]] = arith.remf %[[f0]], %[[f1]] : f32


// Operator precedence

int i3 = 3;
int i4 = 4;
int i5 = 5;
int i6 = 6;

float f3 = 3.0;
float f4 = 4.0;
float f5 = 5.0;
float f6 = 6.0;

i2 = i0 + i1 - i3 * i4 / i5 % i6;

// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[add:.*]] = arith.addi %[[i0]], %[[i1]] : i32
// MLIR: %[[i3:.*]] = oq3.variable_load @i3 : i32
// MLIR: %[[i4:.*]] = oq3.variable_load @i4 : i32
// MLIR: %[[mul:.*]] = arith.muli %[[i3]], %[[i4]] : i32
// MLIR: %[[i5:.*]] = oq3.variable_load @i5 : i32
// MLIR: %[[div:.*]] = arith.divsi %[[mul]], %[[i5]] : i32
// MLIR: %[[i6:.*]] = oq3.variable_load @i6 : i32
// MLIR: %[[rem:.*]] = arith.remsi %[[div]], %[[i6]] : i32
// MLIR: %[[sub:.*]] = arith.subi %[[add]], %[[rem]] : i32
// MLIR: oq3.variable_assign @i2 : i32 = %[[sub]]

i2 = i0 % i1 / i3 * i4 - i5 + i6;

// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[rem:.*]] = arith.remsi %[[i0]], %[[i1]] : i32
// MLIR: %[[i3:.*]] = oq3.variable_load @i3 : i32
// MLIR: %[[div:.*]] = arith.divsi %[[rem]], %[[i3]] : i32
// MLIR: %[[i4:.*]] = oq3.variable_load @i4 : i32
// MLIR: %[[mul:.*]] = arith.muli %[[div]], %[[i4]] : i32
// MLIR: %[[i5:.*]] = oq3.variable_load @i5 : i32
// MLIR: %[[sub:.*]] = arith.subi %[[mul]], %[[i5]] : i32
// MLIR: %[[i6:.*]] = oq3.variable_load @i6 : i32
// MLIR: %[[add:.*]] = arith.addi %[[sub]], %[[i6]] : i32
// MLIR: oq3.variable_assign @i2 : i32 = %[[add]]

i2 = i0 * (i1 - i3);

// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[i3:.*]] = oq3.variable_load @i3 : i32
// MLIR: %[[sub:.*]] = arith.subi %[[i1]], %[[i3]] : i32
// MLIR: %[[mul:.*]] = arith.muli %[[i0]], %[[sub]] : i32
// MLIR: oq3.variable_assign @i2 : i32 = %[[mul]]

f2 = f0 + f1 - f3 * f4 / f5 % f6;

// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[add:.*]] = arith.addf %[[f0]], %[[f1]] : f32
// MLIR: %[[f3:.*]] = oq3.variable_load @f3 : f32
// MLIR: %[[f4:.*]] = oq3.variable_load @f4 : f32
// MLIR: %[[mul:.*]] = arith.mulf %[[f3]], %[[f4]] : f32
// MLIR: %[[f5:.*]] = oq3.variable_load @f5 : f32
// MLIR: %[[div:.*]] = arith.divf %[[mul]], %[[f5]] : f32
// MLIR: %[[f6:.*]] = oq3.variable_load @f6 : f32
// MLIR: %[[rem:.*]] = arith.remf %[[div]], %[[f6]] : f32
// MLIR: %[[sub:.*]] = arith.subf %[[add]], %[[rem]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[sub]]

f2 = f0 % f1 / f3 * f4 - f5 + f6;

// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[rem:.*]] = arith.remf %[[f0]], %[[f1]] : f32
// MLIR: %[[f3:.*]] = oq3.variable_load @f3 : f32
// MLIR: %[[div:.*]] = arith.divf %[[rem]], %[[f3]] : f32
// MLIR: %[[f4:.*]] = oq3.variable_load @f4 : f32
// MLIR: %[[mul:.*]] = arith.mulf %[[div]], %[[f4]] : f32
// MLIR: %[[f5:.*]] = oq3.variable_load @f5 : f32
// MLIR: %[[sub:.*]] = arith.subf %[[mul]], %[[f5]] : f32
// MLIR: %[[f6:.*]] = oq3.variable_load @f6 : f32
// MLIR: %[[add:.*]] = arith.addf %[[sub]], %[[f6]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[add]]

f2 = f0 * (f1 - f3);

// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[f3:.*]] = oq3.variable_load @f3 : f32
// MLIR: %[[sub:.*]] = arith.subf %[[f1]], %[[f3]] : f32
// MLIR: %[[mul:.*]] = arith.mulf %[[f0]], %[[sub]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[mul]]

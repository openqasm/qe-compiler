OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR



int i0 = 1;
int i1 = 2;
int i2;

float f0 = 1.0;
float f1 = 2.0;
float f2;

// Power

i2 = i0 ** i1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=i2, bits=32), right=BinaryOpNode(type=ASTOpTypePow, left=IdentifierNode(name=i0, bits=32), right=IdentifierNode(name=i1, bits=32))
// MLIR: {{.*}} = oq3.variable_load @i0 : i32
// MLIR: {{.*}} = oq3.variable_load @i1 : i32
// MLIR: %[[i2:.*]] = math.ipowi %[[i0:.*]], %[[i1:.*]] : i32

f2 = f0 ** f1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=BinaryOpNode(type=ASTOpTypePow, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=f1, bits=32))
// MLIR: {{.*}} = oq3.variable_load @f0 : f32
// MLIR: {{.*}} = oq3.variable_load @f1 : f32
// MLIR: %[[f2:.*]] = math.powf %[[f0:.*]], %[[f1:.*]] : f32

f2 = f0 ** i1;
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=BinaryOpNode(type=ASTOpTypePow, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=i1, bits=32))
// MLIR: {{.*}} = oq3.variable_load @f0 : f32
// MLIR: {{.*}} = oq3.variable_load @i1 : i32
// MLIR: %[[f2:.*]] = math.fpowi %[[f0:.*]], %[[f1:.*]] : f32

// Cos
f2 = cos(f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeCos, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[val:.*]] = math.cos %[[f0]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Sin
f2 = sin(f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeSin, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[val:.*]] = math.sin %[[f0]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Tan
f2 = tan(f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeTan, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[val:.*]] = math.tan %[[f0]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Not available until LLVM 18
// TODO: Enable the following for LLVM 18

// ArcCos
// f2 = acos(f0);
// DISABLED-AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeArcCos, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// DISABLED-MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// DISABLED-MLIR: %[[val:.*]] = math.acos %[[f0]] : f32
// DISABLED-MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// ArcSin
// f2 = asin(f0);
// DISABLED-AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeArcSin, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// DISABLED-MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// DISABLED-MLIR: %[[val:.*]] = math.asin %[[f0]] : f32
// DISABLED-MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// ArcTan
f2 = atan(f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeArcTan, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[val:.*]] = math.atan %[[f0]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Exp
f2 = exp(f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeExp, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[val:.*]] = math.exp %[[f0]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Ln
f2 = ln(f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeLn, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[val:.*]] = math.log %[[f0]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Sqrt
f2 = sqrt(f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeSqrt, operand=OperandNode(target-identifier=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[val:.*]] = math.sqrt %[[f0]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Unary expression
f2 = cos(f0 + f0);
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f2, bits=32), right=UnaryOpNode(type=ASTOpTypeCos, operand=OperandNode(target=BinaryOpNode(type=ASTOpTypeAdd, left=IdentifierNode(name=f0, bits=32), right=IdentifierNode(name=f0, bits=32))
// MLIR: %[[f0_0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f0_1:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[sum:.*]] = arith.addf %[[f0_0]], %[[f0_1]] : f32
// MLIR: %[[val:.*]] = math.cos %[[sum]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[val]]

// Operator precedence

int i3 = 3;

float f3 = 3.0;

i2 = i0 ** i1 * i3;
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[pow:.*]] = math.ipowi %[[i0]], %[[i1]] : i32
// MLIR: %[[i3:.*]] = oq3.variable_load @i3 : i32
// MLIR: %[[mul:.*]] = arith.muli %[[pow]], %[[i3]] : i32
// MLIR: oq3.variable_assign @i2 : i32 = %[[mul]]


i2 = i0 * i1 ** i3;
// WARNING: Operator precedence of power is incorrect.
// TODO: power should take precedence over multiply.
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[mul:.*]] = arith.muli %[[i0]], %[[i1]] : i32
// MLIR: %[[i3:.*]] = oq3.variable_load @i3 : i32
// MLIR: %[[pow:.*]] = math.ipowi %[[mul]], %[[i3]] : i32
// MLIR: oq3.variable_assign @i2 : i32 = %[[pow]]

i2 = (i0 * i1) ** i3;
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[mul:.*]] = arith.muli %[[i0]], %[[i1]] : i32
// MLIR: %[[i3:.*]] = oq3.variable_load @i3 : i32
// MLIR: %[[pow:.*]] = math.ipowi %[[mul]], %[[i3]] : i32
// MLIR: oq3.variable_assign @i2 : i32 = %[[pow]]


f2 = f0 ** f1 * f3;
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[pow:.*]] = math.powf %[[f0]], %[[f1]] : f32
// MLIR: %[[f3:.*]] = oq3.variable_load @f3 : f32
// MLIR: %[[mul:.*]] = arith.mulf %[[pow]], %[[f3]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[mul]]

f2 = f0 * f1 ** f3;
// WARNING: Operator precedence of power is incorrect.
// TODO: power should take precedence over multiply
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[mul:.*]] = arith.mulf %[[f0]], %[[f1]] : f32
// MLIR: %[[f3:.*]] = oq3.variable_load @f3 : f32
// MLIR: %[[pow:.*]] = math.powf %[[mul]], %[[f3]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[pow]]

f2 = (f0 * f1) ** f3;
// MLIR: %[[f0:.*]] = oq3.variable_load @f0 : f32
// MLIR: %[[f1:.*]] = oq3.variable_load @f1 : f32
// MLIR: %[[mul:.*]] = arith.mulf %[[f0]], %[[f1]] : f32
// MLIR: %[[f3:.*]] = oq3.variable_load @f3 : f32
// MLIR: %[[pow:.*]] = math.powf %[[mul]], %[[f3]] : f32
// MLIR: oq3.variable_assign @f2 : f32 = %[[pow]]

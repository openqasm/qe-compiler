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

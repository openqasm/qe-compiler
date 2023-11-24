OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test0, parameters=[], returns=ResultNode(void))
// MLIR: func private @test0()
extern test0();

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test1, parameters=[DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: ], returns=ResultNode(void))
// MLIR: func private @test1(i32)
extern test1(int a);

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test2, parameters=[], returns=ResultNode(CBitNode(name=bitset, bits=1)))
// MLIR: func private @test2() -> !quir.cbit<1>
extern test2() -> bit;

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test3, parameters=[DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: ], returns=ResultNode(CBitNode(name=bitset, bits=1)))
// MLIR: func private @test3(i32) -> !quir.cbit<1>
extern test3(int a) -> bit;

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test4, parameters=[DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: DeclarationNode(type=ASTTypeFloat, IdentifierNode(name=b, bits=32))
// AST-PRETTY: ], returns=ResultNode(MPDecimalNode(name=ast-mpdecimal-type-param-{{.*}}, bits=32)))
// MLIR: func private @test4(i32, f32) -> f32
extern test4(int a, float b) -> float[32];

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test5, parameters=[DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=a, bits=32))
// AST-PRETTY: ], returns=ResultNode(MPIntegerNode(name=ast-mpinteger-type-param-{{.*}}, value=0, bits=64, signed=1)))
// MLIR: func private @test5(i32) -> i64
extern test5(int[32] a) -> int[64];


// TODO: Requires array support to be added to parser
// extern test6([10]int a) -> [20]int[32];


int a = 10;
bit b;
float c = 10.1;
float[32] d;
int[32] e = 10;
int[64] f;

// AST-PRETTY: FunctionCallNode(type=ASTTypeFunctionCallExpression, expressions=IdentifierNode(name=a, bits=32), kernelDefinition=DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32)
// MLIR: [[RES3:%[0-9]*]] = func.call @test3(%{{[0-9]+}}) : (i32) -> !quir.cbit<1>
// MLIR-NEXT: oq3.variable_assign @b : !quir.cbit<1> = [[RES3]]
b = test3(a);

// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=d, bits=32), right=FunctionCallNode(type=ASTTypeFunctionCallExpression, expressions=IdentifierNode(name=a, bits=32)IdentifierNode(name=c, bits=32), kernelDefinition=DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: DeclarationNode(type=ASTTypeFloat, IdentifierNode(name=b, bits=32))
// AST-PRETTY: ResultNode(MPDecimalNode(name=ast-mpdecimal-type-param-{{.*}}, bits=32)))
// MLIR: [[RES4:%[0-9]*]] = func.call @test4(%{{[0-9]+}}, %{{[0-9]+}}) : (i32, f32) -> f32
// MLIR-NEXT: oq3.variable_assign @d : f32 = [[RES4]]
d = test4(a, c);

// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f, bits=64), right=FunctionCallNode(type=ASTTypeFunctionCallExpression, expressions=IdentifierNode(name=e, bits=32), kernelDefinition=DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=a, bits=32))
// AST-PRETTY: ResultNode(MPIntegerNode(name=ast-mpinteger-type-param-{{.*}}, value=0, bits=64, signed=1)))
// MLIR: [[RES5:%[0-9]*]] = func.call @test5(%{{[0-9]+}}) : (i32) -> i64
// MLIR-NEXT: oq3.variable_assign @f : i64 = [[RES5]]
f = test5(e);

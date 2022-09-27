OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// TODO: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

// NOTE: Only partial support for extern has been added while
// work is awaited on updating the parser version in
// https://github.ibm.com/IBM-Q-Software/vt-dynamic-circuits/issues/962
// we have merged this code as is so that it will be updated in the
// parser update. After the update we will come back and complete extern
// support.

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test0, parameters=[], returns=ResultNode(void))
extern test0();

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test1, parameters=[DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: ], returns=ResultNode(void))
extern test1(int a);

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test2, parameters=[], returns=ResultNode(CBitNode(name=bitset, bits=1)))
extern test2() -> bit;

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test3, parameters=[DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: ], returns=ResultNode(CBitNode(name=bitset, bits=1)))
extern test3(int a) -> bit;

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test4, parameters=[DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: DeclarationNode(type=ASTTypeFloat, IdentifierNode(name=b, bits=32))
// AST-PRETTY: ], returns=ResultNode(MPDecimalNode(name=ast-mpdecimal-type-param-{{.*}}, bits=32)))
extern test4(int a, float b) -> float[32];

// AST-PRETTY: DeclarationNode(type=Unknown, ExternNode(name=test5, parameters=[DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=a, bits=32))
// AST-PRETTY: ], returns=ResultNode(MPIntegerNode(name=ast-mpinteger-type-param-{{.*}}, value=0, bits=64, signed=1)))
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
b = test3(a);

// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=d, bits=32), right=FunctionCallNode(type=ASTTypeFunctionCallExpression, expressions=IdentifierNode(name=a, bits=32)IdentifierNode(name=c, bits=32), kernelDefinition=DeclarationNode(type=ASTTypeInt, IntNode(signed=true, value=10, bits=32))
// AST-PRETTY: DeclarationNode(type=ASTTypeFloat, IdentifierNode(name=b, bits=32))
// AST-PRETTY: ResultNode(MPDecimalNode(name=ast-mpdecimal-type-param-{{.*}}, bits=32)))
d = test4(a, c);

// AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=f, bits=64), right=FunctionCallNode(type=ASTTypeFunctionCallExpression, expressions=IdentifierNode(name=e, bits=32), kernelDefinition=DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=a, bits=32))
// AST-PRETTY: ResultNode(MPIntegerNode(name=ast-mpinteger-type-param-{{.*}}, value=0, bits=64, signed=1)))
f = test5(e);

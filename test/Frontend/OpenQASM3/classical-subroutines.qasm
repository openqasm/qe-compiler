// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
OPENQASM 3.0;

// AST-PRETTY: DeclarationNode(type=ASTTypeFunctionDeclaration, FunctionDeclarationNode(FunctionDefinitionNode(name=classical_params, mangled name=_QF16classical_paramsFrII32EFp0_II32_1iEFp1_II32_1fEE_,
// AST-PRETTY: parameters=[DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=i, bits=32))
// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=f, bits=32))
// AST-PRETTY: ],
// AST-PRETTY: results=ResultNode(MPIntegerNode(name=ast-mpinteger-type-param-{{.*}}, value=0, bits=32, signed=1))
// AST-PRETTY: statements=[
// AST-PRETTY: ReturnNode(BinaryOpNode(type=ASTOpTypeMul, left=IdentifierNode(name=i, bits=32), right=IdentifierNode(name=f, bits=32))
// AST-PRETTY: )])
def classical_params(int[32] i, int[32] f) -> int[32] {
  return i * f;
}

int n = 64;

// AST-PRETTY: DeclarationNode(type=ASTTypeMPInteger, IdentifierNode(name=f2, bits=64))
// AST-PRETTY: results=ResultNode(MPIntegerNode(name=ast-mpinteger-type-param-{{.*}}, value=0, bits=64, signed=1))
// AST-PRETTY: statements=[
// AST-PRETTY: ReturnNode(BinaryOpNode(type=ASTOpTypeMul, left=IdentifierNode(name=i2, bits=64), right=IdentifierNode(name=f2, bits=64))
// AST-PRETTY: )])
def param_input(int[n] i2, int[n] f2) -> int[n] {
  return i2 * f2;
}

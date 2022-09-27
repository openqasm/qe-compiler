// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
OPENQASM 3.0;

// AST-PRETTY: DeclarationNode(type=ASTTypeFunctionDeclaration, FunctionDeclarationNode(FunctionDefinitionNode(name=quantum_params, mangled name=_QF14quantum_paramsFrB1EFp0_QC1_2q0EE_,
// AST-PRETTY: parameters=[DeclarationNode(type=ASTTypeQubitContainer, IdentifierNode(name=q0, bits=1))
// AST-PRETTY: ],
// AST-PRETTY: results=ResultNode(CBitNode(name=bitset, bits=1))
// AST-PRETTY: statements=[
// AST-PRETTY: ReturnNode(MeasureNode(qubits=[QubitContainerNode(QubitNode(name=%q0:0, bits=1))], result=CBitNode(name=ast-measure-result-{{.*}}, bits=1))
// AST-PRETTY: )])
def quantum_params(qubit q0) -> bit {
  return measure q0;
}

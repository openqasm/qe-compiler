OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=a, bits=1))
bit a;
// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=a, bits=1))
// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0
// MLIR: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]])
// MLIR: oq3.assign_cbit_bit @a<1> [0] : i1 = [[MEASURE0]]
a = measure $0;

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$1:0, bits=1)))
qubit $1;
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=b, bits=1, value=1, MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=ast-measure-result-{{.*}}, bits=1))
// AST-PRETTY: ))
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1
// MLIR: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]])
// MLIR: [[MEASURE1_CAST:%.*]] = "oq3.cast"([[MEASURE1]]) : (i1) -> !quir.cbit<1>
// MLIR: oq3.assign_variable @b : !quir.cbit<1> = [[MEASURE1_CAST]]
bit b = measure $1;

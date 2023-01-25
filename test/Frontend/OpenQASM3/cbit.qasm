OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// MLIR-DAG: oq3.declare_variable @c0 : !quir.cbit<1>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c0, bits=1))
bit c0;

// MLIR-DAG: oq3.declare_variable @my_bit : !quir.cbit<1>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=my_bit, bits=1, value=1))
bit my_bit = 1;

// MLIR-DAG: oq3.declare_variable @c : !quir.cbit<3>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c, bits=3))
bit[3] c;

// MLIR-DAG: oq3.declare_variable @my_one_bits : !quir.cbit<2>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=my_one_bits, bits=2, value=11))
bit[2] my_one_bits = "11";

// MLIR-DAG: oq3.declare_variable @my_bitstring : !quir.cbit<10>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=my_bitstring, bits=10, value=1101001010))
bit[10] my_bitstring = "1101001010";

// MLIR-DAG: oq3.declare_variable @result : !quir.cbit<20>
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=result, bits=20))
bit[20] result;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=result, bits=20)[index=6])
// MLIR: %[[QUBIT:.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: %[[MEASUREMENT:.*]] = quir.measure(%[[QUBIT]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.assign_cbit_bit @result<20> [6] : i1 = %[[MEASUREMENT]]
qubit $0;
result[6] = measure $0;

// AST-PRETTY: condition=IdentifierNode(name=my_one_bits, bits=2),
// MLIR: %[[MY_ONE_BITS:.*]] = oq3.use_variable @my_one_bits : !quir.cbit<2>
// MLIR: %{{.*}} = "quir.cast"(%[[MY_ONE_BITS]]) : (!quir.cbit<2>) -> i1
if (my_one_bits) {
    U(3.1415926, 0, 3.1415926) $0;
}

// AST-PRETTY: condition=IdentifierRefNode(name=result[6], IdentifierNode(name=result, bits=20), index=6),
// MLIR: %[[RESULT:.*]] = oq3.use_variable @result : !quir.cbit<20>
// MLIR: %[[LVAL:.*]] = oq3.cbit_extractbit(%[[RESULT]] : !quir.cbit<20>) [6] : i1
// MLIR: scf.if %[[LVAL]] {
if (result[6]) {
    U(3.1415926, 0, 3.1415926) $0;
}

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR
//
// Test that measurement results reach the correct destination, be that single
// classical bits or individual bits in classical bit registers.

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
qubit $0;
qubit $1;

// Single qubit measurement into single qubit
bit clbit;

// MLIR: [[MEASURE1:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.assign_cbit_bit @clbit<1> [0] : i1 = [[MEASURE1]]
clbit = measure $0;

barrier $0;

// Single qubit measurement into a multi-bit classical register
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=clreg, bits=4))
bit[4] clreg;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=1])
// MLIR: [[MEASURE2:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.assign_cbit_bit @clreg<4> [1] : i1 = [[MEASURE2]]
clreg[1] = measure $0;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=0])
// MLIR: [[MEASURE3:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.assign_cbit_bit @clreg<4> [0] : i1 = [[MEASURE3]]
clreg[0] = measure $0;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=clreg, bits=4)[index=3])
// MLIR: [[MEASURE4:%.*]] = quir.measure([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.assign_cbit_bit @clreg<4> [3] : i1 = [[MEASURE4]]
clreg[3] = measure $1;
//

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// MLIR: oq3.declare_variable @c : !quir.cbit<2>

// Define a hadamard gate

// AST: <GateDeclarationNode>
// AST: <HadamardGate>
// AST: <Gate>
// AST: <Name>h</Name>
// AST-PRETTY: GateDeclarationNode(name=h, params=[], qubits=[QubitNode(name=ast-gate-qubit-param-q-0, bits=0)], qcparams=[q],
// AST-PRETTY: ops=[
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
// AST-PRETTY: ,
// AST-PRETTY: ]
// AST-PRETTY: )
// MLIR: func @h(%arg0: !quir.qubit<1>) {
// MLIR: [[a0:%angle[_0-9]*]] = quir.constant #quir.angle<1.57079632679 : !quir.angle<64>>
// MLIR: quir.builtin_U %arg0, [[a0]], {{.*}}, {{.*}} : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: return
gate h q {
    U(1.57079632679, 0.0, 3.14159265359) q;
}

gate cx control, target { }

// MLIR: func @main() -> i32 {
// Qiskit will provide physical qubits
// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
qubit $0;
qubit $1;

// Array of classical bits is fine
// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=c, bits=2))

bit[2] c;

// AST-PRETTY: HGateOpNode(params=[], qubits=[], qcparams=[$0],
// AST-PRETTY: ops=[
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
// MLIR: quir.call_gate @h([[QUBIT0]]) : (!quir.qubit<1>) -> ()
// MLIR: quir.builtin_CX [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>
h $0;
cx $0, $1;

// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$0:0, bits=1))], result=CBitNode(name=c, bits=2)[index=0])
// AST-PRETTY: MeasureNode(qubits=[QubitContainerNode(QubitNode(name=$1:0, bits=1))], result=CBitNode(name=c, bits=2)[index=1])
// MLIR: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @c<2> [0] : i1 = [[MEASURE0]]
// MLIR: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @c<2> [1] : i1 = [[MEASURE1]]
measure $0 -> c[0];
measure $1 -> c[1];

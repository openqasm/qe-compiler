OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR
// RUN: qss-compiler -X=qasm --emit=mlir %s | grep -v OK | qss-compiler -X=mlir - | FileCheck %s --match-full-lines --check-prefix MLIR

// quantum teleportation example

// AST: <GateDeclarationNode>
// AST: <HadamardGate>
// AST: <Gate>
// AST: <Name>h</Name>
// AST: <Identifier>ast-gate-qubit-param-q-0</Identifier>
// AST: <GateQubitName>q</GateQubitName>
// AST-PRETTY: GateDeclarationNode(name=h, params=[], qubits=[QubitNode(name=ast-gate-qubit-param-q-0, bits=0)], qcparams=[q],
// AST-PRETTY: ops=[
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])
// AST-PRETTY: ,
// AST-PRETTY: ]
// AST-PRETTY: )
// MLIR: func @h(%arg0: !quir.qubit<1>) {
gate h q {
    U(1.57079632679, 0, 3.14159265359) q;
}

// AST: <GateDeclarationNode>
// AST: <Gate>
// AST: <Name>z</Name>
// AST: <GateQubitName>q</GateQubitName>
// AST-PRETTY: GateDeclarationNode(name=z, params=[], qubits=[QubitNode(name=ast-gate-qubit-param-q-0, bits=0)], qcparams=[q],
// AST-PRETTY: ops=[
// AST-PRETTY: UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[q])

// AST-PRETTY: ,
// AST-PRETTY: ]
// AST-PRETTY: )
// MLIR: func @z(%arg0: !quir.qubit<1>) {
gate z q {
    U(0, 0, 3.14159265359) q;
}

// MLIR: func @x(%arg0: !quir.qubit<1>) {
gate x q {
    U(3.14159265359, 0, 3.14159265359) q;
}

gate cx control, target { }

// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
qubit $0;
qubit $1;
qubit $2;

bit c0;
bit c1;
bit c2;

reset $0;
reset $1;
reset $2;

U(0.3, 0.2, 0.1) $0;

h $0;
cx $1, $2;

// AST: <Declaration>
// AST: <Identifier>for_0</Identifier>
// AST: <Type>ASTTypeDuration</Type>
// AST-PRETTY: DeclarationNode(type=ASTTypeDuration, DurationNode(duration=30, unit=Nanoseconds, name=for_0))
// AST-PRETTY: DeclarationNode(type=ASTTypeDuration, DurationNode(duration=40, unit=Nanoseconds, name=for_1))
// AST-PRETTY: DeclarationNode(type=ASTTypeDuration, DurationNode(duration=50, unit=Nanoseconds, name=for_2))
// MLIR: [[DURATION0:%.*]] = quir.constant #quir.duration<"30ns" : !quir.duration>
// MLIR: [[DURATION1:%.*]] = quir.constant #quir.duration<"40ns" : !quir.duration>
// MLIR: [[DURATION2:%.*]] = quir.constant #quir.duration<"50ns" : !quir.duration>
duration for_0 = 30ns;
duration for_1 = 40ns;
duration for_2 = 50ns;

// AST: <DelayStatement>
// AST: <Delay>
// AST: <DelayType>ASTTypeDuration</DelayType>
// AST: <Duration>
// AST: <Identifier>for_0</Identifier>
// AST: <Duration>30</Duration>
// AST: <LengthUnit>Nanoseconds</LengthUnit>
// AST: </Duration>
// AST: <IdentifierList>
// AST: <Identifier>
// AST: <Name>$0</Name>
// AST: <Bits>1</Bits>
// AST-PRETTY: DelayStatementNode(DelayNode(duration=for_0, qubit=IdentifierNode(name=$0, bits=1)))
// AST-PRETTY: DelayStatementNode(DelayNode(duration=for_1, qubit=IdentifierNode(name=$1, bits=1)))
// AST-PRETTY: DelayStatementNode(DelayNode(duration=for_2, qubit=IdentifierNode(name=$2, bits=1)))
// MLIR: quir.delay [[DURATION0]], ([[QUBIT0]]) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR: quir.delay [[DURATION1]], ([[QUBIT1]]) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR: quir.delay [[DURATION2]], ([[QUBIT2]]) : !quir.duration, (!quir.qubit<1>) -> ()
delay [for_0] $0;
delay [for_1] $1;
delay [for_2] $2;

// AST: <BarrierNode>
// AST: <IdentifierList>
// AST: <Identifier>
// AST: <Name>$0</Name>
// AST: <Identifier>
// AST: <Name>$1</Name>
// AST: <Identifier>
// AST: <Name>$2</Name>
// AST-PRETTY: BarrierNode(ids=[
// AST-PRETTY: IdentifierNode(name=$0, bits=1),
// AST-PRETTY: IdentifierNode(name=$1, bits=1),
// AST-PRETTY: IdentifierNode(name=$2, bits=1),
// AST-PRETTY: ])
// MLIR: quir.barrier [[QUBIT0]], [[QUBIT1]], [[QUBIT2]] : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

barrier $0, $1, $2;

cx $0, $1;
h $0;
c0 = measure $0;
// MLIR: %[[MVAL:.*]] = quir.measure({{.*}}) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @c0<1> [0] : i1 = %[[MVAL]]
c1 = measure $1;
// MLIR: %[[MVAL:.*]] = quir.measure({{.*}}) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @c1<1> [0] : i1 = %[[MVAL]]

// AST: <GenericGateOpNode>
// AST: <GateOpNode>
// AST: <Gate>
// AST: <Name>z</Name>
// AST: <Opaque>false</Opaque>
// AST: <Qubits>
// AST: <Qubit>
// AST: <Identifier>$2:0</Identifier>
// AST-PRETTY: IfStatementNode(
// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=c0, bits=1), right=IntNode(signed=true, value=1, bits=32))
// AST-PRETTY: statements=
// AST-PRETTY: GateGenericNode(name=z, params=[], qubits=[], qcparams=[$2],
// AST-PRETTY: )
if (c0==1) {
    z $2;
}

// MLIR: scf.if %{{.*}} {
if (c1==1) {
    // MLIR: quir.call_gate @x({{.*}}) : (!quir.qubit<1>) -> ()
    x $2;
}

c2 = measure $2;

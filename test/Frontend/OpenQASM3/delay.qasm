OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

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

// Qiskit will provide physical qubits

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$1:0, bits=1)))
qubit $0;
qubit $1;

// MLIR-CIRCUITS: quir.circuit @circuit_0(%arg0: !quir.duration, %arg1: !quir.qubit<1>, %arg2: !quir.qubit<1>, %arg3: !quir.duration, %arg4: !quir.duration, %arg5: !quir.duration, %arg6: !quir.duration, %arg7: !quir.duration) {
// MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.duration<"5ns" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DeclarationNode(type=ASTTypeDuration, DurationNode(duration=5, unit=Nanoseconds, name=t))
// AST-PRETTY: DelayStatementNode(DelayNode(duration=t, qubit=IdentifierNode(name=$0, bits=1)))
duration t = 5ns;
delay[t] $0;

// MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.duration<"10ns" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=10Nanoseconds, qubit=IdentifierNode(name=$1, bits=1), ))
delay[10ns] $1;
// MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.duration<"20us" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=20Microseconds, qubit=IdentifierNode(name=$1, bits=1), ))
delay[20us] $1;
// MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.duration<"30ms" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=30Milliseconds, qubit=IdentifierNode(name=$1, bits=1), ))
delay[30ms] $1;
// MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.duration<"40dt" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=40DT, qubit=IdentifierNode(name=$1, bits=1), ))
delay[40dt] $1;

// MLIR-NO-CIRCUITS: {{.*}} = quir.constant #quir.duration<"10ns" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}, {{.*}}) : !quir.duration, (!quir.qubit<1>, !quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=10Nanoseconds, qubit=IdentifierNode(name=$0, bits=1), IdentifierNode(name=$1, bits=1), ))
delay [10ns] $0, $1;
// MLIR-CIRCUITS: quir.return


// MLIR-NO-CIRCUITS: {{.*}} = oq3.declare_stretch : !quir.stretch
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.stretch, (!quir.qubit<1>) -> ()
// AST-PRETTY: StretchStatementNode(StretchNode(name=a))
// AST-PRETTY: DelayStatementNode(DelayNode(stretch=StretchNode(name=a), qubit=IdentifierNode(name=$0, bits=1), ))
stretch a;
delay[a] $0;

// MLIR: quir.delay {{.*}}, ({{.*}}, {{.*}}) : !quir.stretch, (!quir.qubit<1>, !quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(stretch=StretchNode(name=a), qubit=IdentifierNode(name=$0, bits=1), IdentifierNode(name=$1, bits=1), ))
delay[a] $0, $1;

//MLIR-CIRCUITS: quir.return
//MLIR-CIRCUITS: func @main() -> i32 {
// MLIR-CIRCUITS: {{.*}} = quir.constant #quir.duration<"5ns" : !quir.duration>
// MLIR-CIRCUITS: {{.*}} = quir.constant #quir.duration<"10ns" : !quir.duration>
// MLIR-CIRCUITS: {{.*}} = quir.constant #quir.duration<"20us" : !quir.duration>
// MLIR-CIRCUITS: {{.*}} = quir.constant #quir.duration<"30ms" : !quir.duration>
// MLIR-CIRCUITS: {{.*}} = quir.constant #quir.duration<"40dt" : !quir.duration>
// MLIR-CIRCUITS: {{.*}} = quir.constant #quir.duration<"10ns" : !quir.duration>
// MLIR-CIRCUITS: quir.call_circuit @circuit_0({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!quir.{{.*}}, !quir.{{.*}}, !quir.{{.*}}, !quir.{{.*}}, !quir.{{.*}}, !quir.{{.*}}, !quir.{{.*}}, !quir.{{.*}}) -> ()
// TODO: Two oq3.declare_stretch statements are generated independent of --enable-circuits
//       This does no harm but might potentially be fixed at some point
//MLIR-CIRCUITS: %3 = oq3.declare_stretch : !quir.stretch
//MLIR-CIRCUITS: quir.call_circuit @circuit_1({{.*}}, {{.*}},{{.*}}) : (!quir.{{.*}}, !quir.{{.*}}, !quir.{{.*}}) -> ()


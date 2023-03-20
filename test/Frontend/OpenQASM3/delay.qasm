OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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


// MLIR: {{.*}} = quir.constant #quir.duration<"5ns" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DeclarationNode(type=ASTTypeDuration, DurationNode(duration=5, unit=Nanoseconds, name=t))
// AST-PRETTY: DelayStatementNode(DelayNode(duration=t, qubit=IdentifierNode(name=$0, bits=1)))
duration t = 5ns;
delay[t] $0;

// MLIR: {{.*}} = quir.constant #quir.duration<"10ns" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=10Nanoseconds, qubit=IdentifierNode(name=$1, bits=1), ))
delay[10ns] $1;
// MLIR: {{.*}} = quir.constant #quir.duration<"20us" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=20Microseconds, qubit=IdentifierNode(name=$1, bits=1), ))
delay[20us] $1;
// MLIR: {{.*}} = quir.constant #quir.duration<"30ms" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=30Milliseconds, qubit=IdentifierNode(name=$1, bits=1), ))
delay[30ms] $1;
// MLIR: {{.*}} = quir.constant #quir.duration<"40dt" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=40DT, qubit=IdentifierNode(name=$1, bits=1), ))
delay[40dt] $1;

// MLIR: {{.*}} = quir.constant #quir.duration<"10ns" : !quir.duration>
// MLIR: quir.delay {{.*}}, ({{.*}}, {{.*}}) : !quir.duration, (!quir.qubit<1>, !quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(duration=10Nanoseconds, qubit=IdentifierNode(name=$0, bits=1), IdentifierNode(name=$1, bits=1), ))
delay [10ns] $0, $1;

// MLIR: {{.*}} = oq3.declare_stretch : !quir.stretch
// MLIR: quir.delay {{.*}}, ({{.*}}) : !quir.stretch, (!quir.qubit<1>) -> ()
// AST-PRETTY: StretchStatementNode(StretchNode(name=a))
// AST-PRETTY: DelayStatementNode(DelayNode(stretch=StretchNode(name=a), qubit=IdentifierNode(name=$0, bits=1), ))
stretch a;
delay[a] $0;

// MLIR: quir.delay {{.*}}, ({{.*}}, {{.*}}) : !quir.stretch, (!quir.qubit<1>, !quir.qubit<1>) -> ()
// AST-PRETTY: DelayStatementNode(DelayNode(stretch=StretchNode(name=a), qubit=IdentifierNode(name=$0, bits=1), IdentifierNode(name=$1, bits=1), ))
delay[a] $0, $1;

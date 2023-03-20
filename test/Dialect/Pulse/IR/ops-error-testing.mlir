// RUN: qss-opt %s -split-input-file -verify-diagnostics

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

// -----

pulse.sequence @sequence0 (%omega1: !quir.angle<32>, %omega2: !quir.angle<32>) {
	// expected-error@+1 {{'oq3.angle_add' op is not valid within a real-time pulse sequence.}}
	%omega3 = oq3.angle_add %omega1, %omega2 : !quir.angle<32>
	pulse.return
}

// -----

pulse.sequence @sequence1 (%omega1: !quir.angle<32>) {
	pulse.call_sequence @x() : () -> ()
	// expected-error@+1 {{'oq3.cast' op is not valid within a real-time pulse sequence.}}
	%f1 = "oq3.cast"(%omega1) : (!quir.angle<32>) -> f64
	pulse.return
}

pulse.sequence @x()

// -----

pulse.sequence @sequence2 (%cond: i1, %mixed : !pulse.mixed_frame, %waveform: !pulse.waveform) {
	// expected-error@+1 {{'scf.yield' op is not valid within a real-time pulse sequence.}}
	scf.if %cond {
        pulse.play(%mixed, %waveform) : (!pulse.mixed_frame, !pulse.waveform)
	}
	pulse.return
}

// -----

pulse.sequence @sequence3 () {
	// expected-error@+1 {{'oq3.declare_variable' op is not valid within a real-time pulse sequence.}}
	oq3.declare_variable @c1 : !quir.cbit<1>
	pulse.return
}

// -----

pulse.sequence @sequence4 () -> i1 {
	%ub = constant 10 : index
	%cmpval = constant 9 : index
	// expected-error@+1 {{'arith.cmpi' op is not valid within a real-time pulse sequence.}}
	%cond = arith.cmpi "eq", %ub, %cmpval : index
	pulse.return
}

// -----

pulse.sequence @sequence5 (%omega1: !quir.angle<32>) {
    // expected-error@+1 {{'quir.call_gate' op is not valid within a real-time pulse sequence.}}
	quir.call_gate @x() : () -> ()
	%f1 = "oq3.cast"(%omega1) : (!quir.angle<32>) -> f64
	pulse.return
}

quir.circuit @x()

// -----

pulse.sequence @sequence6 () {
    // expected-error@+1 {{'pulse.create_port' op is not valid within a real-time pulse sequence.}}
	%d0 = "pulse.create_port"() {uid = "d0"} : () -> !pulse.port
	pulse.return
}

// -----

// verify MLIR kind error is reported when a port is passed to pulse.play rather than a frame
pulse.sequence @sequence7 (%port : !pulse.port, %waveform: !pulse.waveform) {
	// expected-error@+1 {{'pulse.play' op operand #0 must be a frame of reference for interacting with qubits or a frame associated with a port for interacting with qubits, but got '!pulse.port'}}
    pulse.play(%port, %waveform) : (!pulse.port, !pulse.waveform)
	pulse.return
}

// -----

// verify MLIR sequence required error is reported when a pulse.play is not contained in a sequence
func @invalid_sequence_required1(%mf0: !pulse.mixed_frame,%waveform: !pulse.waveform) {
	// expected-error@+1 {{'pulse.play' op expects parent op 'pulse.sequence'}}
    pulse.play(%mf0, %waveform) : (!pulse.mixed_frame, !pulse.waveform)
}

// -----

// verify MLIR sequence required error is reported when a pulse.capture is not contained in a
// sequence
func @invalid_sequence_required2(%mf0: !pulse.mixed_frame) {
	// expected-error@+1 {{'pulse.capture' op expects parent op 'pulse.sequence'}}
    %res0 = pulse.capture(%mf0) : (!pulse.mixed_frame) -> i1
}

// -----

// verify MLIR sequence required error is reported when a pulse.return is not contained in a
// sequence
func @invalid_sequence_required3() {
	%false = arith.constant 0 : i1
	// expected-error@+1 {{'pulse.return' op expects parent op 'pulse.sequence'}}
    pulse.return %false : i1
}

// -----

// verify MLIR sequence required error is reported when a pulse.set_frequency is not contained in a
// sequence
func @invalid_sequence_required4(%frame: !pulse.frame, %frequency : f64) {
	// expected-error@+1 {{'pulse.set_frequency' op expects parent op 'pulse.sequence'}}
    pulse.set_frequency(%frame, %frequency) : (!pulse.frame, f64)
}

// -----

// verify MLIR sequence required error is reported when a pulse.shift_frequency is not contained in a
// sequence
func @invalid_sequence_required5(%frame: !pulse.frame, %frequency : f64) {
	// expected-error@+1 {{'pulse.shift_frequency' op expects parent op 'pulse.sequence'}}
    pulse.shift_frequency(%frame, %frequency) : (!pulse.frame, f64)
}

// -----

// verify MLIR sequence required error is reported when a pulse.set_phase is not contained in a
// sequence
func @invalid_sequence_required4(%frame: !pulse.frame, %phase : f64) {
	// expected-error@+1 {{'pulse.set_phase' op expects parent op 'pulse.sequence'}}
    pulse.set_phase(%frame, %phase) : (!pulse.frame, f64)
}

// -----

// verify MLIR sequence required error is reported when a pulse.shift_phase is not contained in a
// sequence
func @invalid_sequence_required5(%frame: !pulse.frame, %phase : f64) {
	// expected-error@+1 {{'pulse.shift_phase' op expects parent op 'pulse.sequence'}}
    pulse.shift_phase(%frame, %phase) : (!pulse.frame, f64)
}

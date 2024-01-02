// RUN: qss-opt -split-input-file -verify-diagnostics %s

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

func.func @call_circuit_no_matching_cicuit(){
	quir.circuit @circuit_1() {
		quir.return
	}

	// expected-error@+1 {{quir.call_circuit' op 'circuit_2' does not reference a valid circuit}}
	quir.call_circuit @circuit_2 () : () -> ()
	return
}

// -----

quir.circuit @circuit_1(%arg1: !quir.angle<32>) {
	quir.return
}

func.func @call_circuit_call_does_not_match_getOperands(){
	// expected-error@+1 {{'quir.call_circuit' op incorrect number of operands for the callee circuit}}
	quir.call_circuit @circuit_1 () : () -> ()
	return
}

// -----

// expected-note@below {{return type declared here}}
quir.circuit @circuit_1() -> i1 {
	// expected-error@below {{expected 1 result operands}}
	quir.return
}

func.func @call_circuit_call_does_not_match_results(){
	quir.call_circuit @circuit_1 () : () -> ()
	return
}

// -----

quir.circuit @circuit_1(%arg1: !quir.angle<16>) {
	quir.return
}

func.func @call_circuit_call_operand_types_do_not_match(%arg1: i1){
	// expected-error@below {{'quir.call_circuit' op operand type mismatch at index 0}}
	// expected-note@below {{op input types: 'i1'}}
	// expected-note@below {{function operand types: '!quir.angle<16>'}}
	quir.call_circuit @circuit_1 (%arg1) : (i1) -> ()
	return
}

// -----

// expected-note@below {{return type declared here}}
quir.circuit @circuit_1() -> i32 {
	// expected-error@below {{expected 1 result operands}}
	quir.return
}

func.func @call_circuit_call_result_types_do_not_match() {
	quir.call_circuit @circuit_1 () : () -> (i1)
	return
}

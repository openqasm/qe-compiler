// RUN: qss-opt -split-input-file -verify-diagnostics %s

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// -----

func @call_circuit_no_matching_cicuit(){
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

func @call_circuit_call_does_not_match_operands(){
	// expected-error@+1 {{'quir.call_circuit' op incorrect number of operands for the callee circuit}}
	quir.call_circuit @circuit_1 () : () -> ()
	return
}

// -----

quir.circuit @circuit_1() -> i1 {
	quir.return
}

func @call_circuit_call_does_not_match_results(){
	// expected-error@+1 {{'quir.call_circuit' op incorrect number of results for the callee circuit}}
	quir.call_circuit @circuit_1 () : () -> ()
	return
}

// -----

quir.circuit @circuit_1(%arg1: !quir.angle<16>) {
	quir.return
}

func @call_circuit_call_operand_types_do_not_match(%arg1: i1){
	// expected-error@below {{'quir.call_circuit' op operand type mismatch at index 0}}
	// expected-note@below {{op input types: 'i1'}}
	// expected-note@below {{function operand types: '!quir.angle<16>'}}
	quir.call_circuit @circuit_1 (%arg1) : (i1) -> ()
	return
}

// -----

quir.circuit @circuit_1() -> i32 {
	quir.return
}

func @call_circuit_call_result_types_do_not_match() {
	// expected-error@below {{'quir.call_circuit' op result type mismatch at index 0}}
	// expected-note@below {{op result types: 'i1'}}
	// expected-note@below {{function result types: 'i32'}}
	quir.call_circuit @circuit_1 () : () -> (i1)
	return
}

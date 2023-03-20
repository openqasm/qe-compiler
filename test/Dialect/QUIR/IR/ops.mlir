// RUN: qss-opt %s | qss-opt | FileCheck %s
// Verify the printed output can be parsed.
// RUN: qss-opt %s --mlir-print-op-generic | qss-opt | FileCheck %s

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


// TODO: Qubit operands should be separated from classical operands
// CHECK-LABEL: quir.circuit @circuit1(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.angle<32>) -> i1 {
quir.circuit @circuit1 (%q0 : !quir.qubit<1>, %q1 : !quir.qubit<1>, %theta: !quir.angle<32>) -> i1 {

	%a0 = quir.constant #quir.angle<1.57079632679 : !quir.angle<20>>
	%a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
	%a2 = quir.constant #quir.angle<3.14159265359 : !quir.angle<20>>
	quir.builtin_U %q0, %a0, %a1, %a2 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
	quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>

	// Tests opaque gate call
	// TODO: Qubits operands should be separated from classical operands
	// TODO: Is there a need for both gate_call and call_circuit?
	quir.call_gate @rz(%q1, %theta) : (!quir.qubit<1>, !quir.angle<32>) -> ()


	// Test call of another circuit with input qubit and angle values.
	// CHECK: %{{.*}} = quir.call_circuit @circuit2(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<32>) -> (i1, i1)
	%res0, %res1 = quir.call_circuit @circuit2 (%q1, %theta) : (!quir.qubit<1>, !quir.angle<32>) -> (i1, i1)

	// CHECK: quir.return %{{.*}}
	quir.return %res0 : i1
}

// Test returning of circuit value.
// CHECK: quir.circuit @circuit2(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.angle<32>) -> (i1, i1) {
quir.circuit @circuit2 (%q0: !quir.qubit<1>, %omega: !quir.angle<32>) -> (i1, i1) {
	quir.call_gate @rx(%q0, %omega) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0, %res0 : i1, i1
}

// CHECK-LABEL: func @quir_switch(%arg0: i32) -> i32
func @quir_switch (%flag: i32) -> (i32) {
    // CHECK: %[[y:.*]] = quir.switch %arg0 -> (i32)
    %y = quir.switch %flag -> (i32) {
            // CHECK: %[[y_def:.*]] = arith.constant 0 : i32
            %y_def = arith.constant 0 : i32
            // CHECK: quir.yield %[[y_def]] : i32
            quir.yield %y_def : i32
        } [
        4: {
            // CHECK: %[[y_1:.*]] = arith.constant 1 : i32
            %y_1 = arith.constant 1 : i32
            // CHECK: quir.yield
            quir.yield %y_1 : i32
        }
    ]

    // CHECK: %[[qb1:.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %qb1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // CHECK: %[[theta:.*]] = quir.constant #quir.angle<1.000000e-01 : !quir.angle<1>>
    %theta = quir.constant #quir.angle<0.1 : !quir.angle<1>>
    // CHECK: %[[twotheta:.*]] = quir.constant #quir.angle<2.000000e-01 : !quir.angle<1>>
    %twotheta = quir.constant #quir.angle<0.2 : !quir.angle<1>>

    // CHECK: quir.switch %arg0
    quir.switch %flag {
        // CHECK: quir.builtin_U %[[qb1]], %[[theta]], %[[theta]], %[[theta]] : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        quir.builtin_U %qb1, %theta, %theta, %theta : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
    } [
        0: {
            // CHECK: quir.builtin_U %[[qb1]], %[[twotheta]], %[[theta]], %[[theta]] : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
            quir.builtin_U %qb1, %twotheta, %theta, %theta : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        }
        1: {
            // CHECK: quir.builtin_U %[[qb1]], %[[theta]], %[[twotheta]], %[[theta]] : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
            quir.builtin_U %qb1, %theta, %twotheta, %theta : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        }
    ]

    // CHECK: return %[[y]] : i32
    return %y : i32
}

OPENQASM 3;
// RUN: qss-compiler -X=qasm --emit=mlir --enable-circuit %s | FileCheck %s
// RUN: qss-compiler -X=qasm --emit=mlir --enable-circuit --enable-parameters %s | FileCheck %s --check-prefix PARAM
// XFAIL: *

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

// This test case validates that input and output modifiers for variables are
// parsed correctly and are reflected in generated QUIR.

qubit $0;
qubit $2;

gate sx q { }
gate rz(phi) q { }

input angle theta = 3.141;
// PARAM: qcs.declare_parameter @theta_parameter : !quir.angle<64> = #quir.angle<3.141000e+00 : !quir.angle<64>>

reset $0;

sx $0;
rz(theta) $0;
sx $0;

bit b;

b = measure $0;

// CHECK: quir.circuit @circuit_1() {
// CHECK: %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: %1 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>

// PARAM: %2 = qcs.parameter_load @theta_parameter : !quir.angle<64>
// PARAM: oq3.variable_assign @theta : !quir.angle<64> = %2
// PARAM-NOT: oq3.variable_assign @theta : !quir.angle<64> = %angle

// CHECK: oq3.variable_assign @theta : !quir.angle<64> = %angle

// CHECK: quir.return
// CHECK: }

// CHECK: quir.call_circuit @circuit_0() : () -> ()

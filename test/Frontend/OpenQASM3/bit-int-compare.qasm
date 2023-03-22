OPENQASM 3.0;
// RUN: qss-compiler --num-shots=1  %s | FileCheck %s
//
// Test implicit bit to int cast in comparisons.

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

qubit $0;

bit[5] a = "10101";

gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}

x $0;

// Test implicit cast of bit[n] to int
// CHECK:       %{{.*}} = arith.constant 21 : i32
// CHECK-NEXT:  %{{.*}} = "oq3.cast"(%{{.*}}) : (!quir.cbit<5>) -> i32
// CHECK-NEXT:  %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
if(a == 21){
	x $0;
}

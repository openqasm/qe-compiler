OPENQASM 3.0;
// RUN: qss-compiler --num-shots=1  %s | FileCheck %s
//
// Test implicit bit to int cast in comparisons.

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

qubit $0;

bit[5] a = "10101";

gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}

x $0;

// Test implicit cast of bit[n] to int
// CHECK:       %{{.*}} = arith.constant 21 : i32
// CHECK-NEXT:  %{{.*}} = "quir.cast"(%{{.*}}) : (!quir.cbit<5>) -> i32
// CHECK-NEXT:  %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
if(a == 21){
	x $0;
}

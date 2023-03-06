// RUN: qss-compiler -X=mlir %s | FileCheck %s

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

func @main () -> i32 {
    // CHECK: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // qubit %0;
    %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // result = measure %0;
    %zero_ind = arith.constant 0 : index
    %mres = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
    // if (result==1) {
    %one_i1 = arith.constant 1 : i1
    %condition = arith.cmpi "eq", %mres, %one_i1 : i1
    scf.if %condition {
        // U(pi, 0, pi) %0;
        %zero_ang = quir.constant #quir.angle<0.0 : !quir.angle<20>>
        %pi_ang = quir.constant #quir.angle<3.14159 : !quir.angle<20>>
        quir.builtin_U %q0, %pi_ang, %zero_ang, %pi_ang : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
    }
    %zero = arith.constant 0 : i32
    return %zero : i32
}

// RUN: qss-compiler -X=mlir %s | FileCheck %s

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

func.func @main () -> i32 {
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

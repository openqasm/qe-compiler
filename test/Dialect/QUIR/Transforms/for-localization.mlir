// RUN: qss-compiler -X=mlir --classical-only-detection %s | FileCheck %s

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

%q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
%ang = quir.constant #quir.angle<0.00> : !quir.angle<20>
%ang_incr = quir.constant #quir.angle<0.1> : !quir.angle<20>
%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
// CHECK: %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!quir.angle<20>)
%ang_final = scf.for %iv = %lb to %ub step %step
    iter_args(%ang_iter = %ang) -> (!quir.angle<20>) {
    quir.call_gate @RX(%q1, %ang_iter) : (!quir.qubit<1>, !quir.angle<20>) -> ()
    %ang_sum = oq3.angle_add %ang_iter, %ang_incr : !quir.angle<20>
    scf.yield %ang_sum : !quir.angle<20>
}

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

module {
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qa1 = quir.declare_qubit : !quir.qubit<1>
        %qb1 = quir.declare_qubit : !quir.qubit<1>
        %qc1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: quir.reset %{{.*}} : !quir.qubit<1>
        quir.reset %qa1 : !quir.qubit<1>
        quir.reset %qb1 : !quir.qubit<1>
        quir.reset %qc1 : !quir.qubit<1>
        // CHECK: quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
        %res1 = "quir.measure"(%qb1) : (!quir.qubit<1>) -> i1
        // SYNCH: quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
        // SYNCH-NEXT: qcs.synchronize %{{.*}} : (!quir.qubit<1>) -> ()
        return
    }
}

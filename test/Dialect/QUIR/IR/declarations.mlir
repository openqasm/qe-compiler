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

// This test case checks that QUIR declarations can be parsed from
// textual/assembly input.
module {
    func.func @bar() {
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qa1 = quir.declare_qubit : !quir.qubit<1>
        %qb1 = quir.declare_qubit : !quir.qubit<1>
        %qc1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qd1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: %{{.*}} = quir.constant #quir.angle<1.000000e-01 : !quir.angle<1>>
        %theta = quir.constant #quir.angle<0.1 : !quir.angle<1>>
        // CHECK: %{{.*}} = quir.constant #quir.angle<2.000000e-01  : !quir.angle>
        %mu = quir.constant #quir.angle<0.2 : !quir.angle>
        // CHECK %{{.*}} = quir.constant #quir.duration<1.000000e+00 : !quir.duration<ns>>
        %len1 = quir.constant #quir.duration<10.0> : !quir.duration<ns>
        // CHECK %{{.*}} = oq3.declare_stretch : !quir.stretch
        %s1 = "oq3.declare_stretch"() : () -> !quir.stretch
        // CHECK %{{.*}} = oq3.declare_stretch : !quir.stretch
        %s2 = oq3.declare_stretch : !quir.stretch
        oq3.declare_variable { input } @flags : !quir.cbit<32>
        oq3.declare_variable { output } @result : !quir.cbit<1>
        return
    }
}

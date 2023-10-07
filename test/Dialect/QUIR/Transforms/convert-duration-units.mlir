// RUN: qss-compiler -X=mlir --convert-quir-duration-units='units=dt dt-duration=0.1' %s | FileCheck %s --check-prefix=DT


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

// This test verifies store-forwarding and the removal of invisible stores. All
// variable loads must be replaced by forwarded stored values. Then, any
// remaining stores are invisible as the variables have no lifetime beyond this
// program and are to be removed, together with the allocation of variables.

// CHECK-LABEL: func @quir_durations()
func @quir_durations (%arg : i32) {

    %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>

    %duration_dt0 = quir.constant #quir.duration<10.0 : <dt>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+01 : <dt>>
    %duration_dt1 = quir.constant #quir.duration<10.0 : !quir.duration<dt>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+01 : <dt>>
    %duration_s = quir.constant #quir.duration<10.0 : !quir.duration<s>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+02 : <dt>>
    %duration_ms = quir.constant #quir.duration<10.0 : !quir.duration<ms>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+04 : <dt>>
    %duration_us = quir.constant #quir.duration<10.0 : !quir.duration<us>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+07 : <dt>>
    %duration_ns = quir.constant #quir.duration<10.0 : !quir.duration<ns>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+010 : <dt>>
    %duration_ps = quir.constant #quir.duration<10.0 : !quir.duration<ps>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+013 : <dt>>
    %duration_fs = quir.constant #quir.duration<10.0 : !quir.duration<fs>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+016 : <dt>>

    quir.delay %duration_s, (%q0) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_ms, (%q0) : !quir.duration<ms>, (!quir.qubit<1>) -> ()

    quir.call_circuit @circuit0(%q0, %duration_dt0,  %duration_s, %duration_ms) : (!quir.qubit<1>, !quir.duration<dt>, !quir.duration<s>, !quir.duration<ms>) -> (!quir.duration<ms>, !quir.duration<ms>)
    call @func0(%arg, %q0, %duration_dt0,  %duration_s, %duration_ms) : (i32, !quir.qubit<1>, !quir.duration<dt>, !quir.duration<s>, !quir.duration<ms>) -> (i32, !quir.duration<ms>)
    return
}


quir.circuit @circuit0 (%q: !quir.qubit<1>, %duration_dt0: !quir.duration<dt>, %duration_s: !quir.duration<s>, %duration_ms: !quir.duration<ms>) -> (!quir.duration<ms>, !quir.duration<ms>) {
    quir.delay %duration_dt0, (%q) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    quir.delay %duration_s, (%q) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_ms, (%q) : !quir.duration<ms>, (!quir.qubit<1>) -> ()
    quir.return %duration_ms, %duration_ms : !quir.duration<ms>, !quir.duration<ms>
}

func @func0 (%arg: i32, %q: !quir.qubit<1>, %duration_dt0: !quir.duration<dt>, %duration_s: !quir.duration<s>, %duration_ms: !quir.duration<ms>) -> (i32, !quir.duration<ms>) {
    quir.delay %duration_dt0, (%q) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    quir.delay %duration_s, (%q) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_ms, (%q) : !quir.duration<ms>, (!quir.qubit<1>) -> ()
    return %arg, %duration_ms : i32, !quir.duration<ms>
}

// RUN: qss-compiler -X=mlir --convert-quir-duration-units='units=dt dt-timestep=0.1' %s | FileCheck %s --check-prefix=DT
// RUN: qss-compiler -X=mlir --convert-quir-duration-units='units=s dt-timestep=0.1' %s | FileCheck %s --check-prefix=S


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

// This test verifies that the pass --convert-quir-duration-units
// is able to globally convert duration units throughout the IR.

// CHECK-LABEL: func.func @quir_durations()
func.func @quir_durations (%arg : i32) {

    %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>

    // Test a non-duration constant
    %angle = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>
    // DT: {{.*}} = quir.constant #quir.angle<3.1415926535900001 : !quir.angle<64>>

    %duration_dt0 = quir.constant #quir.duration<10.0 : <dt>>
    // DT: [[duration_dt0:%.*]] = quir.constant #quir.duration<1.000000e+01 : <dt>>
    // S: [[duration_dt0:%.*]] = quir.constant #quir.duration<1.000000e+00 : <s>>
    %duration_dt1 = quir.constant #quir.duration<10.0 : !quir.duration<dt>>
    // DT: {{.*}} = quir.constant #quir.duration<1.000000e+01 : <dt>>
    // S: {{.*}} = quir.constant #quir.duration<1.000000e+00 : <s>>
    %duration_s = quir.constant #quir.duration<10.0 : !quir.duration<s>>
    // DT: [[duration_s:%.*]] = quir.constant #quir.duration<1.000000e+02 : <dt>>
    // S: [[duration_s:%.*]] = quir.constant #quir.duration<1.000000e+01 : <s>>
    %duration_ms = quir.constant #quir.duration<10.0 : !quir.duration<ms>>
    // DT: [[duration_ms:%.*]] = quir.constant #quir.duration<0.099999999999999992 : <dt>>
    // S: [[duration_ms:%.*]] = quir.constant #quir.duration<1.000000e-02 : <s>>
    %duration_us = quir.constant #quir.duration<10.0 : !quir.duration<us>>
    // DT: {{.*}} = quir.constant #quir.duration<9.9999999999999995E-8 : <dt>>
    // S: {{.*}} = quir.constant #quir.duration<1.000000e-05 : <s>>
    %duration_ns = quir.constant #quir.duration<10.0 : !quir.duration<ns>>
    // Floating point precision errors. If this is an issue longrun we should move to APFloat.
    // which is an easy change.
    // S: {{.*}} = quir.constant #quir.duration<1.000000e-08 : <s>>

    %duration_ps = quir.constant #quir.duration<10.0 : !quir.duration<ps>>
    %duration_fs = quir.constant #quir.duration<10.0 : !quir.duration<fs>>

    quir.delay %duration_s, (%q0) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    // DT:  quir.delay [[duration_s]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S:  quir.delay [[duration_s]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_ms, (%q0) : !quir.duration<ms>, (!quir.qubit<1>) -> ()
    // DT:  quir.delay [[duration_ms]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S:  quir.delay [[duration_ms]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()


    quir.call_circuit @circuit0(%q0, %duration_dt0,  %duration_s, %duration_ms) : (!quir.qubit<1>, !quir.duration<dt>, !quir.duration<s>, !quir.duration<ms>) -> (!quir.duration<ms>, !quir.duration<ms>)
    // DT: {{.*}}:2 = quir.call_circuit @circuit0({{.*}}, [[duration_dt0]], [[duration_s]], [[duration_ms]]) : (!quir.qubit<1>, !quir.duration<dt>, !quir.duration<dt>, !quir.duration<dt>) -> (!quir.duration<dt>, !quir.duration<dt>)
    // S: {{.*}}:2 = quir.call_circuit @circuit0({{.*}}, [[duration_dt0]], [[duration_s]], [[duration_ms]]) : (!quir.qubit<1>, !quir.duration<s>, !quir.duration<s>, !quir.duration<s>) -> (!quir.duration<s>, !quir.duration<s>)
    call @func0(%arg, %q0, %duration_dt0,  %duration_s, %duration_ms) : (i32, !quir.qubit<1>, !quir.duration<dt>, !quir.duration<s>, !quir.duration<ms>) -> (i32, !quir.duration<ms>)
    // DT: {{.*}}:2 = call @func0({{.*}}, {{.*}}, [[duration_dt0]], [[duration_s]], [[duration_ms]]) : (i32, !quir.qubit<1>, !quir.duration<dt>, !quir.duration<dt>, !quir.duration<dt>) -> (i32, !quir.duration<dt>)
    // S: {{.*}}:2 = call @func0({{.*}}, {{.*}}, [[duration_dt0]], [[duration_s]], [[duration_ms]]) : (i32, !quir.qubit<1>, !quir.duration<s>, !quir.duration<s>, !quir.duration<s>) -> (i32, !quir.duration<s>)
    return
}


quir.circuit @circuit0 (%q: !quir.qubit<1>, %duration_dt0: !quir.duration<dt>, %duration_s: !quir.duration<s>, %duration_ms: !quir.duration<ms>) -> (!quir.duration<ms>, !quir.duration<ms>) {
// DT: quir.circuit @circuit0(%arg0: !quir.qubit<1>, [[duration_dt0:%.*]]: !quir.duration<dt>, [[duration_s:%.*]]: !quir.duration<dt>, [[duration_ms:%.*]]: !quir.duration<dt>) -> (!quir.duration<dt>, !quir.duration<dt>) {
// S: quir.circuit @circuit0(%arg0: !quir.qubit<1>, [[duration_dt0:%.*]]: !quir.duration<s>, [[duration_s:%.*]]: !quir.duration<s>, [[duration_ms:%.*]]: !quir.duration<s>) -> (!quir.duration<s>, !quir.duration<s>) {
    quir.delay %duration_dt0, (%q) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // DT: quir.delay [[duration_dt0]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S: quir.delay [[duration_dt0]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_s, (%q) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    // DT: quir.delay [[duration_s]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S: quir.delay [[duration_s]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_ms, (%q) : !quir.duration<ms>, (!quir.qubit<1>) -> ()
    // DT: quir.delay [[duration_ms]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S: quir.delay [[duration_ms]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.return %duration_ms, %duration_ms : !quir.duration<ms>, !quir.duration<ms>
    // DT: quir.return [[duration_ms]], [[duration_ms]] : !quir.duration<dt>, !quir.duration<dt>
    // S: quir.return [[duration_ms]], [[duration_ms]] : !quir.duration<s>, !quir.duration<s>
}

func.func @func0 (%arg: i32, %q: !quir.qubit<1>, %duration_dt0: !quir.duration<dt>, %duration_s: !quir.duration<s>, %duration_ms: !quir.duration<ms>) -> (i32, !quir.duration<ms>) {
// DT: func.func @func0(%arg0: i32, %arg1: !quir.qubit<1>, [[duration_dt0:%.*]]: !quir.duration<dt>, [[duration_s:%.*]]: !quir.duration<dt>, [[duration_ms:%.*]]: !quir.duration<dt>) -> (i32, !quir.duration<dt>) {
// S: func.func @func0(%arg0: i32, %arg1: !quir.qubit<1>, [[duration_dt0:%.*]]: !quir.duration<s>, [[duration_s:%.*]]: !quir.duration<s>, [[duration_ms:%.*]]: !quir.duration<s>) -> (i32, !quir.duration<s>) {
    quir.delay %duration_dt0, (%q) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // DT: quir.delay [[duration_dt0]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S: quir.delay [[duration_dt0]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_s, (%q) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    // DT: quir.delay [[duration_s]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S: quir.delay [[duration_s]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    quir.delay %duration_ms, (%q) : !quir.duration<ms>, (!quir.qubit<1>) -> ()
    // DT: quir.delay [[duration_ms]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    // S: quir.delay [[duration_ms]], ({{.*}}) : !quir.duration<s>, (!quir.qubit<1>) -> ()
    return %arg, %duration_ms : i32, !quir.duration<ms>
    // S: return {{.*}}, [[duration_ms]] : i32, !quir.duration<s>
}

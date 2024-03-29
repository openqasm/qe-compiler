// RUN: qss-compiler -X=mlir --pulse-remove-unused-arguments %s | FileCheck %s

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

func.func @main() -> i32 {
    %1 = "pulse.create_port"() {uid = "p0"} : () -> !pulse.port
    %3 = "pulse.mix_frame"(%1) {uid = "mf0-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %5 = "pulse.mix_frame"(%1) {uid = "mf1-p0"} : (!pulse.port) -> !pulse.mixed_frame


    // CHECK-NOT: %{{.}}= "pulse.create_port"() {uid = "p1"} : () -> !pulse.port
    // CHECK-NOT: %{{.}}= "pulse.create_port"() {uid = "p2"} : () -> !pulse.port
    %6 = "pulse.create_port"() {uid = "p1"} : () -> !pulse.port
    %7 = "pulse.create_port"() {uid = "p2"} : () -> !pulse.port

    %8 = pulse.call_sequence @seq_0(%1, %3, %5, %6, %7) : (!pulse.port, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.port, !pulse.port) -> i1
    // CHECK: %{{.}} = pulse.call_sequence @seq_0(%{{.}}, %{{.}}) : (!pulse.mixed_frame, !pulse.mixed_frame) -> i1

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
}

pulse.sequence @seq_0(%arg0: !pulse.port,  %arg1: !pulse.mixed_frame, %arg2: !pulse.mixed_frame, %arg3: !pulse.port, %arg4: !pulse.port) -> i1 {
// CHECK: pulse.sequence @seq_0(%arg0: !pulse.mixed_frame, %arg1: !pulse.mixed_frame) -> i1 {
    %c0_i1 = arith.constant 0 : i1
    %c6_i32 = arith.constant 6 : i32
    %c12_i32 = arith.constant 12 : i32
    %c18_i32 = arith.constant 18 : i32


    pulse.delay(%arg1, %c6_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg1, %c12_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg0, %c12_i32) : (!pulse.mixed_frame, i32)

    pulse.delay(%arg2, %c18_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg2, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)


    pulse.delay(%arg1, %c6_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg2, %c18_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg1, %c6_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg2, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)

    pulse.return %c0_i1 : i1
}

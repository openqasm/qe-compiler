// RUN: qss-compiler -X=mlir -pass-pipeline='pulse.sequence(pulse-merge-delay)' %s | FileCheck %s

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

func @main() -> i32 {
    %1 = "pulse.create_port"() {uid = "p0"} : () -> !pulse.port
    %3 = "pulse.mix_frame"(%1) {uid = "mf0-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %5 = "pulse.mix_frame"(%1) {uid = "mf1-p0"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %{{.}} = "pulse.mix_frame"(%{{.}}) {uid = "mf1-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %c5_i32 = arith.constant 5 : i32
    %c10_i32 = arith.constant 10 : i32
    %c15_i32 = arith.constant 15 : i32

    %6 = pulse.call_sequence @seq_0(%3, %5) : (!pulse.mixed_frame, !pulse.mixed_frame) -> i1

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
}

pulse.sequence @seq_0(%arg0: !pulse.mixed_frame, %arg1: !pulse.mixed_frame) -> i1 {
    // CHECK: %c6_i32 = arith.constant 6 : i32
    // CHECK-NOT: %c12_i32 = arith.constant 12 : i32
    // CHECK: %c18_i32 = arith.constant 18 : i32
    // CHECK: %c36_i32 = arith.constant 36 : i32
    %c6_i32 = arith.constant 6 : i32
    %c12_i32 = arith.constant 12 : i32
    %c18_i32 = arith.constant 18 : i32

    // CHECK-NOT: pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay(%arg0, %c12_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg0, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg1, %c36_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg0, %c12_i32) : (!pulse.mixed_frame, i32)

    pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)

    pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg0, %c6_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg1, %c18_i32) : (!pulse.mixed_frame, i32)

    %c0_i1 = arith.constant 0 : i1
    pulse.return %c0_i1 : i1
}

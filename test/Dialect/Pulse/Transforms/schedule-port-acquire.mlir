// RUN: qss-compiler -X=mlir --pulse-schedule-port %s | FileCheck %s

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

module @acquire_0 attributes {quir.nodeId = 7 : i32, quir.nodeType = "acquire", quir.physicalIds = [0 : i32, 1 : i32, 2 : i32, 3 : i32]} {
  pulse.sequence @seq_0(%arg0: !pulse.mixed_frame, %arg1: !pulse.mixed_frame, %arg2: !pulse.mixed_frame, %arg3: !pulse.mixed_frame, %arg4: !pulse.mixed_frame) -> i1 {
    // CHECK: pulse.sequence @seq_0(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: !pulse.mixed_frame,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]: !pulse.mixed_frame,
    // CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]: !pulse.mixed_frame,
    // CHECK-SAME: %[[ARG3:[A-Za-z0-9]+]]: !pulse.mixed_frame,
    // CHECK-SAME: %[[ARG4:[A-Za-z0-9]+]]: !pulse.mixed_frame) -> i1 {
    // CHECK-NOT: %c800_i32 = arith.constant 800 : i32
    %c800_i32 = arith.constant 800 : i32
    // CHECK-NOT: %c4496_i32 = arith.constant 4496 : i32
    %c4496_i32 = arith.constant 4496 : i32
    // CHECK-NOT: %c4336_i32 = arith.constant 4336 : i32
    %c4336_i32 = arith.constant 4336 : i32
    // CHECK-NOT: %c160_i32 = arith.constant 160 : i32
    // CHECK-NOT: %c13600_i32 = arith.constant 13600 : i32
    %c13600_i32 = arith.constant 13600 : i32
    // CHECK-NOT: %c18096_i32 = arith.constant 18096 : i32
    %c18096_i32 = arith.constant 18096 : i32
    // CHECK-NOT: pulse.delay(%arg0, %c4496_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg0, %c4496_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay {pulse.timepoint = 0 : i64}(%[[ARG1]], %c4336_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay {pulse.timepoint = 4336 : i64}(%[[ARG0]], %c160_i32) : (!pulse.mixed_frame, i32)
    // CHECK: %0 = pulse.capture {pulse.timepoint = 4496 : i64}(%[[ARG0]]) : (!pulse.mixed_frame) -> i1
    // CHECK-NOT: pulse.delay {pulse.timepoint = 4496 : i64}(%{{.*}}, %c13600_i32) : (!pulse.mixed_frame, i32)
    %0 = pulse.capture(%arg0) : (!pulse.mixed_frame) -> i1
    pulse.delay(%arg0, %c800_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg1, %c4336_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg1, %c13600_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg2, %c18096_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg3, %c18096_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg4, %c18096_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay {pulse.timepoint = 4496 : i64}(%{{.*}}, %c18096_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay {pulse.timepoint = 4496 : i64}(%{{.*}}, %c18096_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay {pulse.timepoint = 4496 : i64}(%{{.*}}, %c18096_i32) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.return {pulse.timepoint = 18096 : i64} %{{.*}} : i1
    pulse.return %0 : i1
  }
  pulse.sequence @seq_1(%arg0: !pulse.mixed_frame, %arg1: !pulse.mixed_frame, %arg2: !pulse.mixed_frame, %arg3: !pulse.mixed_frame, %arg4: !pulse.mixed_frame) -> (i1, i1) {
    %c1000_i32 = arith.constant 1000 : i32
    pulse.delay(%arg0, %c1000_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay(%[[ARG1]], %c1000_i32) : (!pulse.mixed_frame, i32)
    %0 = pulse.call_sequence @seq_0(%arg0, %arg1, %arg2, %arg3, %arg4) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> i1
    // CHECK: %0 = pulse.call_sequence @seq_0(%arg0, %arg1, %arg2, %arg3, %arg4) {pulse.duration = 18096 : i64, pulse.timepoint = 1000 : i64}
    pulse.delay(%arg0, %c1000_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay(%[[ARG1]], %c1000_i32) : (!pulse.mixed_frame, i32)
    %1 = pulse.call_sequence @seq_0(%arg0, %arg1, %arg2, %arg3, %arg4) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> i1
    // CHECK: %1 = pulse.call_sequence @seq_0(%arg0, %arg1, %arg2, %arg3, %arg4) {pulse.duration = 18096 : i64, pulse.timepoint = 20096 : i64} 
    pulse.delay(%arg0, %c1000_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay(%[[ARG1]], %c1000_i32) : (!pulse.mixed_frame, i32)
    pulse.return %0, %1 : i1, i1
  }
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i32 = arith.constant 0 : i32
    %2 = "pulse.create_port"() {uid = "p0"} : () -> !pulse.port
    %4 = "pulse.mix_frame"(%2) {uid = "mf0-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %6 = "pulse.mix_frame"(%2) {uid = "mf1-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %7 = "pulse.create_port"() {uid = "p1"} : () -> !pulse.port
    %9 = "pulse.mix_frame"(%7) {uid = "mf0-p1"} : (!pulse.port) -> !pulse.mixed_frame
    %10 = "pulse.create_port"() {uid = "p2"} : () -> !pulse.port
    %12 = "pulse.mix_frame"(%10) {uid = "mf0-p2"} : (!pulse.port) -> !pulse.mixed_frame
    %13 = "pulse.create_port"() {uid = "p3"} : () -> !pulse.port
    %15 = "pulse.mix_frame"(%13) {uid = "mf0-p3"} : (!pulse.port) -> !pulse.mixed_frame
    %16 = pulse.call_sequence @seq_0(%4, %6, %9, %12, %15) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> i1
    // CHECK: {{.*}} = pulse.call_sequence @seq_0(%1, %2, %4, %6, %8) {pulse.duration = 18096 : i64}
    %17:2 = pulse.call_sequence @seq_1(%4, %6, %9, %12, %15) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> (i1,i1)
    // CHECK: {{.*}}:2 = pulse.call_sequence @seq_1(%1, %2, %4, %6, %8) {pulse.duration = 39192 : i64}
    return %c0_i32 : i32
  }
}

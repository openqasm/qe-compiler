// RUN: qss-compiler -X=mlir --pulse-schedule-port-module %s | FileCheck %s
module @drive_0 attributes {quir.nodeId = 0 : i32, quir.nodeType = "drive", quir.physicalId = 0 } {
  pulse.sequence @seq_0(%arg0: !pulse.waveform, %arg1: !pulse.waveform, %arg2: !pulse.mixed_frame, %arg3: !pulse.mixed_frame) -> i1 {
    // CHECK: pulse.sequence @seq_0(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: !pulse.waveform,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]: !pulse.waveform,
    // CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]: !pulse.mixed_frame,
    // CHECK-SAME: %[[ARG3:[A-Za-z0-9]+]]: !pulse.mixed_frame) -> i1 {
    %false = arith.constant 0: i1
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %c10_i32 = arith.constant 10 : i32
    // CHECK: %false = arith.constant false
    // CHECK-NOT: %c2_i32 = arith.constant 2 : i32
    // CHECK-NOT: %c3_i32 = arith.constant 3 : i32
    // CHECK-NOT: %c7_i32 = arith.constant 7 : i32
    // CHECK-NOT: %c10_i32 = arith.constant 10 : i32
    pulse.delay(%arg2, %c2_i32) : (!pulse.mixed_frame, i32)
    pulse.play(%arg2, %arg0) : (!pulse.mixed_frame, !pulse.waveform)
    pulse.delay(%arg2, %c10_i32) : (!pulse.mixed_frame, i32)
    pulse.delay(%arg3, %c7_i32) : (!pulse.mixed_frame, i32)
    pulse.play(%arg3, %arg1) : (!pulse.mixed_frame, !pulse.waveform)
    pulse.delay(%arg3, %c3_i32) : (!pulse.mixed_frame, i32)
    // CHECK-NOT: pulse.delay {pulse.timepoint = 0 : i64}(%[[ARG2]], %{{.*}}) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.play {pulse.timepoint = 2 : i64}(%[[ARG2]], %[[ARG0]]) : (!pulse.mixed_frame, !pulse.waveform)
    // CHECK-NOT: pulse.delay {pulse.timepoint = 5 : i64}(%[[ARG3]], %{{.*}}) : (!pulse.mixed_frame, i32)
    // CHECK: pulse.play {pulse.timepoint = 7 : i64}(%[[ARG3]], %[[ARG1]]) : (!pulse.mixed_frame, !pulse.waveform)
    // CHECK: pulse.return {pulse.timepoint = 15 : i64} %false : i1
    pulse.return %false : i1
  }
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i32 = arith.constant 0 : i32
    %2 = "pulse.create_port"() {uid = "p0"} : () -> !pulse.port
    %4 = "pulse.mix_frame"(%2) {uid = "mf0-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %6 = "pulse.mix_frame"(%2) {uid = "mf1-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %7 = pulse.create_waveform dense<[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0] ]> : tensor<3x2xf64> -> !pulse.waveform
    %8 = pulse.create_waveform dense<[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0] ]> : tensor<5x2xf64> -> !pulse.waveform
    %9 = pulse.call_sequence @seq_0(%7, %8, %4, %6) : (!pulse.waveform, !pulse.waveform, !pulse.mixed_frame, !pulse.mixed_frame) -> i1
    return %c0_i32 : i32
  }
}

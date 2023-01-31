// RUN: qss-compiler -X=mlir --pulse-schedule-port-module %s | FileCheck %s
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
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %0 = complex.create %cst, %cst : complex<f64>
    %1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
    %2 = "pulse.create_port"() {uid = "Q0"} : () -> !pulse.port
    %3 = pulse.create_frame(%0, %cst, %1) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %4 = "pulse.mix_frame"(%2, %3) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %5 = pulse.create_frame(%0, %cst, %1) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %6 = "pulse.mix_frame"(%2, %5) {signalType = "drive"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %7 = "pulse.create_port"() {uid = "Q1"} : () -> !pulse.port
    %8 = pulse.create_frame(%0, %cst, %1) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %9 = "pulse.mix_frame"(%7, %8) {signalType = "drive"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %10 = "pulse.create_port"() {uid = "Q2"} : () -> !pulse.port
    %11 = pulse.create_frame(%0, %cst, %1) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %12 = "pulse.mix_frame"(%10, %11) {signalType = "drive"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %13 = "pulse.create_port"() {uid = "Q3"} : () -> !pulse.port
    %14 = pulse.create_frame(%0, %cst, %1) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %15 = "pulse.mix_frame"(%13, %14) {signalType = "drive"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %16 = pulse.call_sequence @seq_0(%4, %6, %9, %12, %15) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> i1
    return %c0_i32 : i32
  }
}

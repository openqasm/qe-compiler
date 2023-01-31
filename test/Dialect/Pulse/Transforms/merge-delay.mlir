// RUN: qss-compiler -X=mlir -pass-pipeline='pulse.sequence(pulse-merge-delay)' %s | FileCheck %s
func @main() -> i32 {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = complex.create %cst, %cst : complex<f64>
    %angle = quir.constant #quir.angle<0.000000e+00 : !quir.angle<20>>
    %1 = "pulse.create_port"() {uid = "Q0"} : () -> !pulse.port
    %2 = pulse.create_frame(%0, %cst, %angle) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %3 = "pulse.mix_frame"(%1, %2) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %4 = pulse.create_frame(%0, %cst, %angle) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %5 = "pulse.mix_frame"(%1, %4) {signalType = "drive"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    // CHECK: %{{.}} = "pulse.mix_frame"(%{{.}}, %{{.}}) {signalType = "drive"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
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

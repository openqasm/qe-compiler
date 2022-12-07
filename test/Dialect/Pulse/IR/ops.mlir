// RUN: qss-opt %s | qss-opt | FileCheck %s
// Verify the printed output can be parsed.
// RUN: qss-opt %s --mlir-print-op-generic | qss-opt | FileCheck %s

func @main () {
    %d0 = "pulse.create_port"() {uid = "d0"} : () -> !pulse.port
    // CHECK: %[[D0:.*]] = "pulse.create_port"() {uid = "d0"} : () -> !pulse.port
    %m0 = "pulse.create_port"() {uid = "m0"} : () -> !pulse.port
    // CHECK: %[[M0:.*]] = "pulse.create_port"() {uid = "m0"} : () -> !pulse.port

    %amp_i = arith.constant 0.0012978777572167797 : f64
    %amp_q = arith.constant 0.0012978777572167797 : f64
    %amp = complex.create %amp_i, %amp_q : complex<f64>
    %freq = arith.constant 100.e6 : f64
    %phase = quir.constant #quir.angle<0.0 : !quir.angle<20>>
    %f0 = "pulse.create_frame"(%amp, %freq, %phase) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    // CHECK: %[[F0:.*]] = pulse.create_frame(%{{.*}}, %{{.*}}, %{{.*}}) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame

    %mf0 = "pulse.mix_frame"(%d0, %f0) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    // CHECK: %[[MF0:.*]] = "pulse.mix_frame"(%[[D0]], %[[F0]]) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame

    %param_amp_i = arith.constant 0.10086211860780928 : f64
    %param_amp_j = arith.constant 0.0012978777572167797 : f64
    %param_amp = complex.create %param_amp_i, %param_amp_j : complex<f64>
    // CHECK: %[[AMP:.*]] = complex.create %{{.*}}, %{{.*}} : complex<f64>

    %res0, %res1 = pulse.call_sequence @test_pulse_ops (%d0, %m0, %f0, %mf0, %param_amp) : (!pulse.port, !pulse.port, !pulse.frame, !pulse.mixed_frame, complex<f64>) -> (i1, i1)
    // CHECK: %{{.*}}:2 = pulse.call_sequence @test_pulse_ops(%[[D0]], %[[M0]], %[[F0]], %[[MF0]], %[[AMP]]) : (!pulse.port, !pulse.port, !pulse.frame, !pulse.mixed_frame, complex<f64>) -> (i1, i1)

    return
}

pulse.sequence @test_pulse_ops (%d0: !pulse.port, %m0: !pulse.port, %f0: !pulse.frame, %mf0: !pulse.mixed_frame,%amp: complex<f64>) -> (i1, i1) {
// CHECK: pulse.sequence @test_pulse_ops(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: !pulse.port,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]: !pulse.port,
    // CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]: !pulse.frame,
    // CHECK-SAME: %[[ARG3:[A-Za-z0-9]+]]: !pulse.mixed_frame,
    // CHECK-SAME: %[[ARG4:[A-Za-z0-9]+]]: complex<f64>) -> (i1, i1) {

    %duration = arith.constant 160 : i32
    %sigma = arith.constant 40 : i32
    %width = arith.constant 1000 : i32
    %beta = arith.constant -1.3677586253287046 : f64
    %gaussian_waveform = pulse.gaussian(%duration, %amp, %sigma) : (i32, complex<f64>, i32) -> !pulse.waveform
    // CHECK %{{.*}} = pulse.gaussian(%c160_i32, %[[ARG3]], %c40_i32) : (i32, complex<f64>, i32) -> !pulse.waveform
    %gaussian_square_waveform = pulse.gaussian_square(%duration, %amp, %sigma, %width) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
    // CHECK %{{.*}} = pulse.gaussian_square(%c160_i32, %[[ARG3]], %c40_i32, %c1000_i32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
    %drag_waveform = pulse.drag(%duration, %amp, %sigma, %beta) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
    // CHECK %[[DRAG:.*]] = pulse.drag(%c160_i32, %[[ARG3]], %c40_i32, %cst) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
    %const_waveform = pulse.const_waveform(%duration, %amp) : (i32, complex<f64>) -> !pulse.waveform
    // CHECK %{{.*}} = pulse.const_waveform(%c160_i32, %[[ARG3]]) : (i32, complex<f64>) -> !pulse.waveform
    %kernel_waveform = pulse.create_waveform dense<[[0.0, 0.5], [0.5, 0.5], [0.5, 0.0]]> : tensor<3x2xf64> -> !pulse.waveform
    // CHECK %[[KERNELWAVEFORM:.*]] = pulse.create_waveform dense<[[0.000000e+00, 5.000000e-01], [5.000000e-01, 5.000000e-01], [5.000000e-01, 0.000000e+00]]> : tensor<3x2xf64> -> !pulse.waveform

    %mf1 = "pulse.mix_frame"(%d0, %f0) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    // CHECK: %{{.*}}   = "pulse.mix_frame"(%[[ARG0]], %[[ARG2]]) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame

    %param_amp_i = arith.constant 0.10086211860780928 : f64
    %param_amp_j = arith.constant 0.0012978777572167797 : f64
    %param_amp = complex.create %param_amp_i, %param_amp_j : complex<f64>
    // CHECK: %[[AMP:.*]] = complex.create %{{.*}}, %{{.*}} : complex<f64>

    %fc = arith.constant 200.e04 : f64

    pulse.set_frequency(%mf0, %fc) : (!pulse.mixed_frame, f64)
    // CHECK pulse.set_frequency(%[[ARG0]], %cst_0) : (!pulse.mixed_frame, f64)
    pulse.set_frequency(%f0, %fc) : (!pulse.frame, f64)
    // CHECK pulse.set_frequency(%[[ARG2]], %cst_0) : (!pulse.frame, f64)

    pulse.shift_frequency(%mf0, %fc) : (!pulse.mixed_frame, f64)
    // CHECK pulse.shift_frequency(%[[ARG0]], %cst_0) : (!pulse.mixed_frame, f64)
    pulse.shift_frequency(%f0, %fc) : (!pulse.frame, f64)
    // CHECK pulse.shift_frequency(%[[ARG2]], %cst_0) : (!pulse.frame, f64)

    %angle = arith.constant 3.14 : f64
    pulse.set_phase(%mf0, %angle) : (!pulse.mixed_frame, f64)
    // CHECK pulse.set_phase(%[[ARG0]], %cst_1) : (!pulse.mixed_frame, f64)
    pulse.set_phase(%f0, %angle) : (!pulse.frame, f64)
    // CHECK pulse.set_phase(%[[ARG2]], %cst_1) : (!pulse.frame, f64)

    pulse.shift_phase(%mf0, %angle) : (!pulse.mixed_frame, f64)
    // CHECK pulse.shift_phase(%[[ARG0]], %cst_1) : (!pulse.mixed_frame, f64)
    pulse.shift_phase(%f0, %angle) : (!pulse.frame, f64)
    // CHECK pulse.shift_phase(%[[ARG2]], %cst_1) : (!pulse.frame, f64)

    pulse.barrier %f0 : !pulse.frame
    // CHECK pulse.barrier %[[ARG2]] : !pulse.frame

    %delay_duration = arith.constant 100 : i32
    pulse.delay(%mf0, %delay_duration) : (!pulse.mixed_frame, i32)
    // CHECK pulse.delay(%arg0, %c100_i32) : (!pulse.mixed_frame, i32)

    pulse.play(%mf0, %drag_waveform) : (!pulse.mixed_frame, !pulse.waveform)
    // CHECK pulse.play(%[[ARG0]], %2) : (!pulse.mixed_frame, !pulse.waveform)

    %kernel = pulse.create_kernel(%kernel_waveform) : (!pulse.waveform) -> !pulse.kernel
    // CHECK %[[KERNEL:.*]] = pulse.create_kernel(%[[KERNELWAVEFORM]]) : (!pulse.waveform) -> !pulse.kernel
    %res0 = pulse.capture(%mf0) : (!pulse.mixed_frame) -> i1
    // CHECK %[[RESULT:.*]] = pulse.capture(%[[ARG1]]) : (!pulse.mixed_frame) -> i1

    pulse.return %res0, %res0: i1, i1
    // CHECK pulse.return %[[RESULT]], %[[RESULT]] : i1, i1
}
// CHECK: }

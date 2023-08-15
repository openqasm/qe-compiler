module {
  func @main() -> (i1, i1) {
    %0 = "pulse.create_port"() {uid = "p0"} : () -> !pulse.port
    %1 = "pulse.create_port"() {uid = "p1"} : () -> !pulse.port
    %2 = "pulse.create_frame"() {uid = "f0"} : () -> !pulse.frame
    %3 = "pulse.mix_frame"(%0) {uid = "mf0-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %cst = arith.constant 0.10086211860780928 : f64
    %cst_0 = arith.constant 0.0012978777572167797 : f64
    %4 = complex.create %cst, %cst_0 : complex<f64>
    %5:2 = pulse.call_sequence @test_pulse_ops(%0, %1, %2, %3, %4) : (!pulse.port, !pulse.port, !pulse.frame, !pulse.mixed_frame, complex<f64>) -> (i1, i1)
    return %5#0, %5#1 : i1, i1
  }
  pulse.sequence public @test_pulse_ops(%arg0: !pulse.port, %arg1: !pulse.port, %arg2: !pulse.frame, %arg3: !pulse.mixed_frame, %arg4: complex<f64>) -> (i1, i1) {
    %c160_i32 = arith.constant 160 : i32
    %c40_i32 = arith.constant 40 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %cst = arith.constant -1.3677586253287046 : f64
    %0 = pulse.gaussian(%c160_i32, %arg4, %c40_i32) : (i32, complex<f64>, i32) -> !pulse.waveform
    %1 = pulse.gaussian_square(%c160_i32, %arg4, %c40_i32, %c1000_i32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
    %2 = pulse.drag(%c160_i32, %arg4, %c40_i32, %cst) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
    %3 = pulse.const_waveform(%c160_i32, %arg4) : (i32, complex<f64>) -> !pulse.waveform
    %4 = pulse.create_waveform dense<[[0.000000e+00, 5.000000e-01], [5.000000e-01, 5.000000e-01], [5.000000e-01, 0.000000e+00]]> : tensor<3x2xf64> -> !pulse.waveform
    %5 = "pulse.mix_frame"(%arg0) {uid = "mf1-p0"} : (!pulse.port) -> !pulse.mixed_frame
    %cst_0 = arith.constant 0.10086211860780928 : f64
    %cst_1 = arith.constant 0.0012978777572167797 : f64
    %6 = complex.create %cst_0, %cst_1 : complex<f64>
    %cst_2 = arith.constant 2.000000e+06 : f64
    pulse.set_frequency(%arg3, %cst_2) : (!pulse.mixed_frame, f64)
    pulse.set_frequency(%arg2, %cst_2) : (!pulse.frame, f64)
    pulse.shift_frequency(%arg3, %cst_2) : (!pulse.mixed_frame, f64)
    pulse.shift_frequency(%arg2, %cst_2) : (!pulse.frame, f64)
    %cst_3 = arith.constant 3.140000e+00 : f64
    pulse.set_phase(%arg3, %cst_2) : (!pulse.mixed_frame, f64)
    pulse.set_phase(%arg2, %cst_2) : (!pulse.frame, f64)
    pulse.shift_phase(%arg3, %cst_2) : (!pulse.mixed_frame, f64)
    pulse.shift_phase(%arg2, %cst_2) : (!pulse.frame, f64)
    pulse.barrier %arg2 : !pulse.frame
    %c100_i32 = arith.constant 100 : i32
    pulse.delay(%arg3, %c100_i32) : (!pulse.mixed_frame, i32)
    pulse.play(%arg3, %2) : (!pulse.mixed_frame, !pulse.waveform)
    %7 = pulse.create_kernel(%4) : (!pulse.waveform) -> !pulse.kernel
    %8 = pulse.capture(%arg3) : (!pulse.mixed_frame) -> i1
    pulse.return %8, %8 : i1, i1
  }
}

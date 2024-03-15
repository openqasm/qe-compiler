module {
  pulse.sequence public @test() -> i1 {
    %c5_i32 = arith.constant 5 : i32
    %cst = arith.constant 2.500000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = complex.create %cst, %cst_0 : complex<f64>
    %1 = pulse.const_waveform(%c5_i32, %0) : (i32, complex<f64>) -> !pulse.waveform
    %false = arith.constant false
    pulse.return %false : i1
  }
}

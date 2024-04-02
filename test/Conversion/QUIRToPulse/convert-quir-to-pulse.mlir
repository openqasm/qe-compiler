// RUN: qss-compiler %s --quir-to-pulse | FileCheck %s

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

module {
  quir.circuit @circuit_0_q5_q3_circuit_1_q5(%arg0: !quir.qubit<1> {quir.physicalId = 5 : i32}, %arg1: !quir.qubit<1> {quir.physicalId = 3 : i32}) -> (i1, i1) attributes {quir.classicalOnly = false, quir.physicalIds = [3 : i32, 5 : i32]} {
  // CHECK-NOT: quir.circuit @circuit_0_q5_q3_circuit_1_q5(%arg0: !quir.qubit<1> {quir.physicalId = 5 : i32}, %arg1: !quir.qubit<1> {quir.physicalId = 3 : i32}) -> (i1, i1) attributes {quir.classicalOnly = false, quir.physicalIds = [3 : i32, 5 : i32]} {
    quir.call_gate @x(%arg1) {pulse.calName = "x_3"} : (!quir.qubit<1>) -> ()
    quir.call_gate @sx(%arg0) {pulse.calName = "sx_5"} : (!quir.qubit<1>) -> ()
    %0:2 = quir.measure(%arg1, %arg0) {pulse.calName = "measure_3_5"} : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
    quir.return %0#0, %0#1 : i1, i1
  }
  quir.circuit @circuit_2_q5_q3_circuit_3_q5(%arg0: !quir.qubit<1> {quir.physicalId = 5 : i32}, %arg1: !quir.qubit<1> {quir.physicalId = 3 : i32}) -> (i1, i1) attributes {quir.classicalOnly = false, quir.physicalIds = [3 : i32, 5 : i32]} {
  // CHECK-NOT: quir.circuit @circuit_2_q5_q3_circuit_3_q5(%arg0: !quir.qubit<1> {quir.physicalId = 5 : i32}, %arg1: !quir.qubit<1> {quir.physicalId = 3 : i32}) -> (i1, i1) attributes {quir.classicalOnly = false, quir.physicalIds = [3 : i32, 5 : i32]} {
    %angle = quir.constant #quir.angle<1.5707963267948966> : !quir.angle<64>
    quir.call_gate @rz(%arg0, %angle) {pulse.calName = "rz_5"} : (!quir.qubit<1>, !quir.angle<64>) -> ()
    quir.call_gate @sx(%arg0) {pulse.calName = "sx_5"} : (!quir.qubit<1>) -> ()
    quir.call_gate @rz(%arg0, %angle) {pulse.calName = "rz_5"} : (!quir.qubit<1>, !quir.angle<64>) -> ()
    quir.builtin_CX {pulse.calName = "cx_5_3"} %arg0, %arg1 : !quir.qubit<1>, !quir.qubit<1>
    %0:2 = quir.measure(%arg1, %arg0) {pulse.calName = "measure_3_5"} : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
    quir.return %0#0, %0#1 : i1, i1
  }
  pulse.sequence @x_3(%arg0: !pulse.mixed_frame) -> i1 attributes {pulse.argPorts = ["q3-drive-port"], pulse.args = ["q3-drive-mixframe"]} {
    %x3_pulse = pulse.create_waveform {pulse.waveformName = "x3_pulse"} dense<[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]> : tensor<16x2xf64> -> !pulse.waveform
    pulse.play {pulse.duration = 16 : i64, pulse.timepoint = 0 : i64}(%arg0, %x3_pulse) : (!pulse.mixed_frame, !pulse.waveform)
    %false = arith.constant false
    pulse.return %false : i1
  }
  pulse.sequence @measure_3_5(%arg0: !pulse.mixed_frame, %arg2: !pulse.mixed_frame,%arg4: !pulse.mixed_frame, %arg6: !pulse.mixed_frame) -> (i1, i1)
  attributes {pulse.argPorts = ["q3-readout-port", "q3-capture-port", "q5-readout-port", "q5-capture-port"],
  pulse.args = ["q3-readout-mixframe", "q3-capture-mixframe", "q5-readout-mixframe", "q5-capture-mixframe"]} {
    %q3_readout = pulse.create_waveform {pulse.waveformName = "q3_readout_pulse"} dense<[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]> : tensor<16x2xf64> -> !pulse.waveform
    %q5_readout = pulse.create_waveform {pulse.waveformName = "q5_readout_pulse"} dense<[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]> : tensor<16x2xf64> -> !pulse.waveform
    pulse.play {pulse.duration = 16 : i64, pulse.timepoint = 0 : i64}(%arg0, %q3_readout) : (!pulse.mixed_frame, !pulse.waveform)
    pulse.play {pulse.duration = 16 : i64, pulse.timepoint = 0 : i64}(%arg4, %q5_readout) : (!pulse.mixed_frame, !pulse.waveform)
    %0 = pulse.capture {pulse.duration = 1024 : i64, pulse.timepoint = 272 : i64}(%arg2) : (!pulse.mixed_frame) -> i1
    %1 = pulse.capture {pulse.duration = 1024 : i64, pulse.timepoint = 272 : i64}(%arg6) : (!pulse.mixed_frame) -> i1
    pulse.return %0, %1 : i1, i1
  }
  pulse.sequence @rz_5(%arg0: f64, %arg1: !pulse.mixed_frame) -> i1
  attributes {pulse.argPorts = ["", "q5-drive-port"], pulse.args = ["angle", "q5-drive-mixframe"]} {
    pulse.shift_phase {pulse.timepoint = 0 : i64}(%arg1, %arg0) : (!pulse.mixed_frame, f64)
    %false = arith.constant false
    pulse.return %false : i1
  }
  pulse.sequence @sx_5(%arg0: !pulse.mixed_frame) -> i1
  attributes {pulse.argPorts = ["q5-drive-port"], pulse.args = ["q5-drive-mixframe"]} {
    %sx5_pulse = pulse.create_waveform {pulse.waveformName = "sx5_pulse"} dense<[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]> : tensor<16x2xf64> -> !pulse.waveform
    pulse.play {pulse.duration = 16 : i64, pulse.timepoint = 0 : i64}(%arg0, %sx5_pulse) : (!pulse.mixed_frame, !pulse.waveform)
    %false = arith.constant false
    pulse.return %false : i1
  }
  pulse.sequence @cx_5_3(%arg0: !pulse.mixed_frame, %arg1: !pulse.mixed_frame) -> i1
  attributes {pulse.argPorts = ["q3-drive-port", "q5-drive-port"],
  pulse.args = ["q3-5-cx-mixframe", "q5-3-cx-mixframe"]} {
    %cx_5_3_pulse = pulse.create_waveform {pulse.waveformName = "cx_5_3_pulse"} dense<[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]> : tensor<16x2xf64> -> !pulse.waveform
    %cst_0 = arith.constant 1.5707963267948966 : f64
    pulse.shift_phase {pulse.timepoint = 0 : i64}(%arg0, %cst_0) : (!pulse.mixed_frame, f64)
    pulse.play {pulse.duration = 16 : i64, pulse.timepoint = 0 : i64}(%arg1, %cx_5_3_pulse) : (!pulse.mixed_frame, !pulse.waveform)
    %false = arith.constant false
    pulse.return %false : i1
  }
  // CHECK: pulse.sequence @circuit_0_q5_q3_circuit_1_q5_sequence(%arg0: !pulse.mixed_frame, %arg1: !pulse.mixed_frame, %arg2: !pulse.mixed_frame, %arg3: !pulse.mixed_frame, %arg4: !pulse.mixed_frame, %arg5: !pulse.mixed_frame) -> (i1, i1, i1, i1) {
    // CHECK: %0 = pulse.call_sequence @x_3(%arg0) : (!pulse.mixed_frame) -> i1
    // CHECK: %1 = pulse.call_sequence @sx_5(%arg1) : (!pulse.mixed_frame) -> i1
    // CHECK: %2:2 = pulse.call_sequence @measure_3_5(%arg2, %arg3, %arg4, %arg5) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> (i1, i1)
    // CHECK: pulse.return %0, %1, %2#0, %2#1 : i1, i1, i1, i1

  // CHECK: pulse.sequence @circuit_2_q5_q3_circuit_3_q5_sequence(%arg0: !pulse.mixed_frame, %arg1: !pulse.mixed_frame, %arg2: !pulse.mixed_frame, %arg3: !pulse.mixed_frame, %arg4: !pulse.mixed_frame, %arg5: !pulse.mixed_frame, %arg6: !pulse.mixed_frame) -> (i1, i1, i1, i1, i1, i1) {
    // CHECK: %cst = arith.constant 1.5707963267948966 : f64
    // CHECK: %0 = pulse.call_sequence @rz_5(%cst, %arg0) : (f64, !pulse.mixed_frame) -> i1
    // CHECK: %1 = pulse.call_sequence @sx_5(%arg0) : (!pulse.mixed_frame) -> i1
    // CHECK: %2 = pulse.call_sequence @rz_5(%cst, %arg0) : (f64, !pulse.mixed_frame) -> i1
    // CHECK: %3 = pulse.call_sequence @cx_5_3(%arg1, %arg2) : (!pulse.mixed_frame, !pulse.mixed_frame) -> i1
    // CHECK: %4:2 = pulse.call_sequence @measure_3_5(%arg3, %arg4, %arg5, %arg6) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> (i1, i1)
    // CHECK: pulse.return %0, %1, %2, %3, %4#0, %4#1 : i1, i1, i1, i1, i1, i1

  func.func @main() -> i32 attributes {quir.classicalOnly = false} {
    // CHECK: %0 = "pulse.create_port"() {uid = "q3-drive-port"} : () -> !pulse.port
    // CHECK: %1 = "pulse.mix_frame"(%0) {uid = "q3-drive-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %2 = "pulse.create_port"() {uid = "q5-drive-port"} : () -> !pulse.port
    // CHECK: %3 = "pulse.mix_frame"(%2) {uid = "q5-drive-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %4 = "pulse.create_port"() {uid = "q3-readout-port"} : () -> !pulse.port
    // CHECK: %5 = "pulse.mix_frame"(%4) {uid = "q3-readout-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %6 = "pulse.create_port"() {uid = "q3-capture-port"} : () -> !pulse.port
    // CHECK: %7 = "pulse.mix_frame"(%6) {uid = "q3-capture-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %8 = "pulse.create_port"() {uid = "q5-readout-port"} : () -> !pulse.port
    // CHECK: %9 = "pulse.mix_frame"(%8) {uid = "q5-readout-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %10 = "pulse.create_port"() {uid = "q5-capture-port"} : () -> !pulse.port
    // CHECK: %11 = "pulse.mix_frame"(%10) {uid = "q5-capture-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %12 = "pulse.mix_frame"(%0) {uid = "q3-5-cx-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    // CHECK: %13 = "pulse.mix_frame"(%2) {uid = "q5-3-cx-mixframe"} : (!pulse.port) -> !pulse.mixed_frame
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      qcs.shot_init {qcs.num_shots = 1000 : i32}
      %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      // CHECK-NOT: %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      // CHECK-NOT: %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      // CHECK-NOT: %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      %3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
      // CHECK-NOT: %3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
      %4 = quir.declare_qubit {id = 4 : i32} : !quir.qubit<1>
      // CHECK-NOT: %4 = quir.declare_qubit {id = 4 : i32} : !quir.qubit<1>
      %5 = quir.declare_qubit {id = 5 : i32} : !quir.qubit<1>
      // CHECK-NOT: %5 = quir.declare_qubit {id = 5 : i32} : !quir.qubit<1>
      %7:2 = quir.call_circuit @circuit_0_q5_q3_circuit_1_q5(%5, %3) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK-NOT: %7:2 = quir.call_circuit @circuit_0_q5_q3_circuit_1_q5(%5, %3) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK: %14:4 = pulse.call_sequence @circuit_0_q5_q3_circuit_1_q5_sequence(%1, %3, %5, %7, %9, %11) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> (i1, i1, i1, i1)
      quir.barrier %3, %5 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
      // CHECK-NOT: %quir.barrier %3, %5 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
      %8:2 = quir.call_circuit @circuit_2_q5_q3_circuit_3_q5(%5, %3) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK-NOT: %8:2 = quir.call_circuit @circuit_2_q5_q3_circuit_3_q5(%5, %3) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK: %15:6 = pulse.call_sequence @circuit_2_q5_q3_circuit_3_q5_sequence(%3, %12, %13, %5, %7, %9, %11) : (!pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame, !pulse.mixed_frame) -> (i1, i1, i1, i1, i1, i1)
    } {qcs.shot_loop}
    return %c0_i32 : i32
  }
}

// RUN: qss-compiler -X=mlir --pulse-remove-unused-arguments %s | FileCheck %s

// (C) Copyright IBM 2023.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

func @main() -> i32 {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = complex.create %cst, %cst : complex<f64>
    %angle = quir.constant #quir.angle<0.000000e+00 : !quir.angle<20>>
    %1 = "pulse.create_port"() {uid = "Q0"} : () -> !pulse.port
    %2 = pulse.create_frame(%0, %cst, %angle) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %3 = "pulse.mix_frame"(%1, %2) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %4 = pulse.create_frame(%0, %cst, %angle) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %5 = "pulse.mix_frame"(%1, %4) {signalType = "drive"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame


    // CHECK-NOT: %{{.}}= "pulse.create_port"() {uid = "Q1"} : () -> !pulse.port
    // CHECK-NOT: %{{.}}= "pulse.create_port"() {uid = "Q2"} : () -> !pulse.port
    %6 = "pulse.create_port"() {uid = "Q1"} : () -> !pulse.port
    %7 = "pulse.create_port"() {uid = "Q2"} : () -> !pulse.port

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

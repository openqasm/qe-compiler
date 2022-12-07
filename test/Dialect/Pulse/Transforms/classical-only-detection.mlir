// RUN: qss-compiler -X=mlir --pulse-classical-only-detection %s | FileCheck %s

// based on quir version of the classical-only-detection test

// first validate that the pulse classical-only-detection does not harm the quir pass
// determinations

func private @kernel1(memref<1xi1>, memref<1xi1>, memref<1xi1>) -> memref<1xi1> attributes {quir.classicalOnly = true}
// CHECK: func private @kernel1(memref<1xi1>, memref<1xi1>, memref<1xi1>) -> memref<1xi1>
// CHECK-SAME: attributes {quir.classicalOnly = true}
func @subroutine1 (%ang1 : !quir.angle<20>, %ang2 : !quir.angle<20>, %q1 : !quir.qubit<1>, %q2 : !quir.qubit<1>) -> (!quir.cbit<1>) attributes {quir.classicalOnly = false} {
    // CHECK: func @subroutine1
    // CHECK: attributes {quir.classicalOnly = false} {
    %zero = arith.constant 0 : index
    %ang3 = quir.angle_add %ang1, %ang2 : !quir.angle<20>
    %ang4 = quir.constant #quir.angle<0.9 : !quir.angle<20>>
    %f1 = "quir.cast"(%ang3) : (!quir.angle<20>) -> f64
    %f2 = "quir.cast"(%ang4) : (!quir.angle<20>) -> f64
    %cond1 = arith.cmpf "ogt", %f1, %f2 : f64
    // CHECK: {quir.classicalOnly = false}
    scf.if %cond1 {
        %cond2 = arith.cmpf "oeq", %f1, %f2 : f64
        // CHECK: {quir.classicalOnly = false}
        scf.if %cond2 {
            "quir.call_gate"(%q1) {callee = @Z} : (!quir.qubit<1>) -> ()
        } else {
            "quir.call_gate"(%q1) {callee = @X} : (!quir.qubit<1>) -> ()
        } {quir.classicalOnly = false}
    }  {quir.classicalOnly = false}
    %cond2 = arith.cmpf "oeq", %f1, %f2 : f64
    // CHECK: {quir.classicalOnly = false}
    scf.if %cond2 {
        "quir.call_gate"(%q1) {callee = @X} : (!quir.qubit<1>) -> ()
    } {quir.classicalOnly = false}
    %mres1 = "quir.measure"(%q2) : (!quir.qubit<1>) -> i1
    %c1 = "quir.cast"(%mres1) : (i1) -> !quir.cbit<1>
    return %c1 : !quir.cbit<1>
}

// test pulse classical only detection
// next add pulse.sequence and  validate that the pulse classical-only-detection
// labels the sequence as quir.classicalOnly = false

func private @kernel2(memref<1xi1>, memref<1xi1>, memref<1xi1>) -> memref<1xi1>
// CHECK: func private @kernel2(memref<1xi1>, memref<1xi1>, memref<1xi1>) -> memref<1xi1>
// CHECK-SAME: attributes {quir.classicalOnly = true}
func @subroutine2 () {
    // CHECK: func @subroutine2()
    // CHECK-SAME: attributes {quir.classicalOnly = false} {

    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %0 = complex.create %cst, %cst : complex<f64>
    %1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
    %2 = "pulse.create_port"() {uid = "Q0"} : () -> !pulse.port
    %3 = pulse.create_frame(%0, %cst, %1) : (complex<f64>, f64, !quir.angle<20>) -> !pulse.frame
    %4 = "pulse.mix_frame"(%2, %3) {signalType = "measure"} : (!pulse.port, !pulse.frame) -> !pulse.mixed_frame
    %5 = pulse.create_waveform dense<[[0.0, 1.0]]> : tensor<1x2xf64> -> !pulse.waveform
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %cond1 = arith.cmpf "ogt", %zero, %one : f64

    %c2 = arith.constant 0 : index
    %c3 = arith.constant 10 : index
    %c4 = arith.constant 1 : index
    // CHECK: {quir.classicalOnly = false}
    scf.for %arg0 = %c2 to %c3 step %c4 {
        // CHECK: {quir.classicalOnly = false}
        scf.if %cond1 {
            %cond2 = arith.cmpf "oeq", %zero, %one : f64
            // CHECK: {quir.classicalOnly = false}
            scf.if %cond2 {
                pulse.play(%4, %5) : (!pulse.mixed_frame, !pulse.waveform)
            } else {
                %6 = pulse.call_sequence @seq_0(%5, %4) : (!pulse.waveform, !pulse.mixed_frame) -> i1
            }
        }
    }

    return
}
pulse.sequence @seq_0(%arg1: !pulse.waveform, %arg2: !pulse.mixed_frame) -> i1 {
// CHECK: pulse.sequence
// CHECK-SAME: {quir.classicalOnly = false} {
    %c281_i32 = arith.constant 281 : i32
    %c321_i32 = arith.constant 321 : i32
    pulse.delay(%arg2, %c281_i32) : (!pulse.mixed_frame, i32)
    pulse.play(%arg2, %arg1) : (!pulse.mixed_frame, !pulse.waveform)
    %0 = pulse.capture(%arg2) : (!pulse.mixed_frame) -> i1
    pulse.delay(%arg2, %c321_i32) : (!pulse.mixed_frame, i32)
    pulse.return %0 : i1
}

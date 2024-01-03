// RUN: qss-compiler -X=mlir --classical-only-detection %s | FileCheck %s

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

func.func private @kernel1 (%ca1 : memref<1xi1>, %ca2 : memref<1xi1>, %ca3 : memref<1xi1>) -> memref<1xi1>
func.func private @kernel2 (memref<?xi1>) -> memref<1xi1>
func.func @subroutine1 (%ang1 : !quir.angle<20>, %ang2 : !quir.angle<20>, %q1 : !quir.qubit<1>, %q2 : !quir.qubit<1>) -> (!quir.cbit<1>) {
    %zero = arith.constant 0 : index
    %ang3 = oq3.angle_add %ang1, %ang2 : !quir.angle<20>
    %ang4 = quir.constant #quir.angle<0.9> : !quir.angle<20>
    %f1 = "oq3.cast"(%ang3) : (!quir.angle<20>) -> f64
    %f2 = "oq3.cast"(%ang4) : (!quir.angle<20>) -> f64
    %cond1 = arith.cmpf "ogt", %f1, %f2 : f64
// CHECK: {quir.classicalOnly = false}
    scf.if %cond1 {
        %cond2 = arith.cmpf "oeq", %f1, %f2 : f64
// CHECK: {quir.classicalOnly = false}
        scf.if %cond2 {
            "quir.call_gate"(%q1) {callee = @Z} : (!quir.qubit<1>) -> ()
        } else {
            "quir.call_gate"(%q1) {callee = @X} : (!quir.qubit<1>) -> ()
        }
    }
    %cond2 = arith.cmpf "oeq", %f1, %f2 : f64
// CHECK: {quir.classicalOnly = false}
    scf.if %cond2 {
        "quir.call_gate"(%q1) {callee = @X} : (!quir.qubit<1>) -> ()
    }
    %mres1 = "quir.measure"(%q2) : (!quir.qubit<1>) -> i1
    %c1 = "oq3.cast"(%mres1) : (i1) -> !quir.cbit<1>
    return %c1 : !quir.cbit<1>
}
func.func private @proto (%qa1 : !quir.qubit<1>) -> ()
func.func @gateCall1(%q1 : !quir.qubit<1>, %lambda : !quir.angle<1>) -> () {
    %zero = quir.constant #quir.angle<0.0> : !quir.angle<1>
    quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
    "quir.call_gate"(%q1) {callee = @proto} : (!quir.qubit<1>) -> ()
    return
}
func.func @gateCall2(%q1 : !quir.qubit<1>, %lambda : !quir.angle) -> () {
    %zero = quir.constant #quir.angle<0.0> : !quir.angle
    quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle, !quir.angle, !quir.angle
    return
}
func.func @multiQubitGateCall(%qa1 : !quir.qubit<1>, %qb1 : !quir.qubit<1>) -> () {
    quir.builtin_CX %qa1, %qb1 : !quir.qubit<1>, !quir.qubit<1>
    return
}
func.func @defcalGate2(%qa1 : !quir.qubit<1> {quir.qubit_id = 0 : i32}, %theta : !quir.angle {quir.value = 0.5 : f64}) -> () {
    %zero = quir.constant #quir.angle<0.0> : !quir.angle
    quir.builtin_U %qa1, %zero, %theta, %zero : !quir.qubit<1>, !quir.angle, !quir.angle, !quir.angle
    return
}
func.func @defcalRX(%qa1 : !quir.qubit<1>, %theta : !quir.angle<20>) -> () {
    quir.call_gate @proto1(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<20>) -> ()
    return
}
func.func @defcalRX_q0(%qa1 : !quir.qubit<1> {quir.qubit_id = 0 : i32}, %theta : !quir.angle<20>) -> () attributes {quir.orig_func_name = "defcalRX"} {
    quir.call_gate @proto2(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<20>) -> ()
    return
}
func.func @defcalRX_q0_api2(%qa1 : !quir.qubit<1> {quir.qubit_id = 0 : i32}, %theta : !quir.angle<20> {quir.value = 0.5 : f64}) -> ()
    attributes {quir.orig_func_name = "defcalRX"}
{
    quir.call_gate @proto3(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<20>) -> ()
    return
}
%ang1 = quir.constant #quir.angle<0.15> : !quir.angle<20>
%ang2 = quir.constant #quir.angle<0.2> : !quir.angle<20>
%q1 = quir.declare_qubit {id = 4 : i32} : !quir.qubit<1>
%q2 = quir.declare_qubit {id = 5 : i32} : !quir.qubit<1>

%zero = arith.constant 0 : index
%ang3 = oq3.angle_add %ang1, %ang2 : !quir.angle<20>
%ang4 = quir.constant #quir.angle<0.9> : !quir.angle<20>
%f1 = "oq3.cast"(%ang3) : (!quir.angle<20>) -> f64
%f2 = "oq3.cast"(%ang4) : (!quir.angle<20>) -> f64
%cond1 = arith.cmpf "ogt", %f1, %f2 : f64
// CHECK: {quir.classicalOnly = false}
scf.if %cond1 {
    %cond2 = arith.cmpf "oeq", %f1, %f2 : f64
// CHECK: {quir.classicalOnly = false}
    scf.if %cond2 {
        "quir.call_gate"(%q1) {callee = @Z} : (!quir.qubit<1>) -> ()
    } else {
        "quir.call_gate"(%q1) {callee = @X} : (!quir.qubit<1>) -> ()
    }
}
%cond2 = arith.cmpf "oeq", %f1, %f2 : f64
// CHECK: {quir.classicalOnly = false}
scf.if %cond2 {
    "quir.call_gate"(%q1) {callee = @X} : (!quir.qubit<1>) -> ()
}
%zero_ind = arith.constant 0 : index
%mres1 = "quir.measure"(%q2) : (!quir.qubit<1>) -> i1
%cb1 = "oq3.cast"(%mres1) : (i1) -> !quir.cbit<1>
oq3.declare_variable @c1 : !quir.cbit<1>
oq3.variable_assign @c1 : !quir.cbit<1> = %cb1
//%res2 = func.call @subroutine1(%ang1, %ang2, %q1, %q2) : (!quir.angle<20>, !quir.angle<20>, !quir.qubit<1>, !quir.qubit<1>) -> memref<1xi1>

%c11 = arith.constant 0 : i1
%c21 = arith.constant 1 : i1
%o11 = arith.constant 10 : i32
%o21 = arith.constant 11 : i32
%o31 = arith.constant 13 : i32
// CHECK: {quir.classicalOnly = true}
%r11 = scf.if %c11 -> (i32) {
    %res3 = arith.addi %o11, %o21 : i32
    scf.yield %res3 : i32
} else {
    %res3 = arith.addi %o11, %o31 : i32
    scf.yield %res3 : i32
}
%out = arith.addi %r11, %o21 : i32

// qubit q;
// qubit r;
// angle[3] c = 0;
%qq1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
%qr1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
%angleC = quir.constant #quir.angle<0.0> : !quir.angle<3>
// reset q;
// reset r;
quir.reset %qq1 : !quir.qubit<1>
quir.reset %qr1 : !quir.qubit<1>
// h r;
"quir.call_gate"(%qr1) {callee = @H} : (!quir.qubit<1>) -> ()
// uint[3] power = 1;
%power1 = arith.constant 1 : i3
%lb = arith.constant 0 : index
%ub = arith.constant 2 : index
%step = arith.constant 1 : index
// for i in [0: 2] {
%c_res, %p_res = scf.for %iv = %lb to %ub step %step
    iter_args(%angleC_iter = %angleC, %power1_iter = %power1) -> (!quir.angle<3>, i3) {
//  reset q;
    quir.reset %qq1 : !quir.qubit<1>
//   h q;
    "quir.call_gate"(%qq1) {callee = @H} : (!quir.qubit<1>) -> ()
//   cphase(power*3*pi/8) q, r;
    %power1_angle = "oq3.cast"(%power1_iter) : (i3) -> !quir.angle<3>
    %angle_multiplicand = quir.constant #quir.angle<0.375> : !quir.angle<3>
    %angleP = oq3.angle_mul %power1_angle, %angle_multiplicand : !quir.angle<3>
    "quir.call_gate"(%qq1, %qr1, %angleP) {callee = @cphase} : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<3>) -> ()
    %angle_zero = quir.constant #quir.angle<0.0> : !quir.angle<3>
    %negC = oq3.angle_sub %angle_zero, %angleC_iter : !quir.angle<3>
//   phase(-c) q;
//   h q;
//   measure q -> c[0];
//   c <<= 1;
//   power <<= 1;
    %angleC_shift = quir.constant #quir.angle<0.0> : !quir.angle<3>
    %power_shift = arith.constant 1 : i3
// }
    scf.yield %angleC_shift, %power_shift : !quir.angle<3>, i3
}
%ub2 = arith.constant 20 : index
%init_sum = arith.constant 0 : i32
%final_sum = scf.for %iv = %lb to %ub2 step %step
    iter_args(%sum_iter = %init_sum) -> (i32) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %sum_body = arith.addi %sum_iter, %iv_i32 : i32
    scf.yield %sum_body : i32
}

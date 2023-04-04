// RUN: qss-compiler -X=mlir %s | FileCheck %s

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
    func private @proto (%qa1 : !quir.qubit<1>) -> ()
    // CHECK-LABEL: func @gateCall1
    func @gateCall1(%q1 : !quir.qubit<1>, %lambda : !quir.angle<1>) -> () {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle<1>>
        // CHECK: quir.builtin_U %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        return
    }
    func @subroutine1(%q1 : !quir.qubit<1>, %phi : !quir.angle, %ub : index) {
        %lb = arith.constant 0 : index
        %step = arith.constant 1 : index
        scf.for %iv = %lb to %ub step %step {
            quir.call_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
            quir.call_defcal_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
            %res = quir.call_defcal_measure @defcalMeasure_q0(%q1,%phi) : (!quir.qubit<1>, !quir.angle) -> i1
            scf.if %res {
                // CHECK: quir.call_gate @defcalPhase_q0(%arg1, %arg0) : (!quir.angle, !quir.qubit<1>) -> ()
                // CHECK: quir.call_gate @defcalPhase_q0(%arg1, %arg0) : (!quir.angle, !quir.qubit<1>) -> ()
                quir.call_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
                quir.call_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
            }
        }
        return
    }
    // CHECK-LABEL: func @bar()
    func @bar() {
        // CHECK: qcs.init
        qcs.init
        // CHECK: qcs.finalize
        qcs.finalize
        // CHECK: qcs.shot_init
        qcs.shot_init
        %0 = arith.constant 1 : i32
        %val = arith.constant 1 : i1
        // quir.constant canonical form example with angle attribute
        // CHECK: %angle{{.*}} = quir.constant #quir.angle<1.000000e+00 : !quir.angle<10>>
        %angle1 = "quir.constant"() {"value" = #quir.angle<1.0 : !quir.angle<10>>} : () -> (!quir.angle<10>)
        // CHECK: %angle{{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<10>>
        %angle2 = quir.constant #quir.angle<0.0 : !quir.angle<10>>
        // CHECK: %angle{{.*}} = quir.constant #quir.angle<1.000000e+01 : !quir.angle>
        %angle3 = quir.constant #quir.angle<10.0 : !quir.angle>
        // CHECK: %angle{{.*}} = quir.constant #quir.angle<3.141591 : !quir.angle<20>>
        %angle4 = quir.constant #quir.angle<3.141591 : !quir.angle<20>>
        // arbitrary constants can also be produced
        // CHECK: %qcst{{.*}} = quir.constant 2 : i16
        %qcst = quir.constant 2 : i16
        // CHECK: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
        %qa1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
        %qb1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
        %qc1 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
        // CHECK: quir.reset %{{.*}} : !quir.qubit<1>
        quir.reset %qc1 : !quir.qubit<1>
        // CHECK: quir.builtin_CX %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.qubit<1>
        quir.builtin_CX %qa1, %qb1 : !quir.qubit<1>, !quir.qubit<1>
        // CHECK: %{{.*}} = quir.constant #quir.angle<1.000000e-01 : !quir.angle<1>>
        %theta = quir.constant #quir.angle<0.1 : !quir.angle<1>>
        // CHECK: quir.builtin_U %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        quir.builtin_U %qb1, %theta, %theta, %theta : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        // CHECK: quir.call_gate @gateCall1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall1} : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // CHECK: quir.call_gate @gateCall1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall1} : (!quir.qubit<1>, !quir.angle<1>) -> ()
        %res10 = "quir.measure"(%qb1) : (!quir.qubit<1>) -> i1
        // CHECK: qcs.synchronize %{{.*}} : (!quir.qubit<1>) -> ()
        qcs.synchronize %qb1 : (!quir.qubit<1>) -> ()
        // CHECK: qcs.synchronize %{{.*}} %{{.*}} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
        qcs.synchronize %qa1, %qb1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
        // CHECK: %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
        %mm1:2 = quir.measure(%qa1, %qb1) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
        // another form with individually named outputs:
        %mm1a, %mm1b = quir.measure(%qa1, %qb1) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
        "quir.call_gate"(%qb1) {callee = @proto} : (!quir.qubit<1>) -> ()
        quir.call_gate @proto(%qb1) : (!quir.qubit<1>) -> ()
        // CHECK: quir.barrier %{{.*}} : (!quir.qubit<1>) -> ()
        "quir.barrier"(%qb1) : (!quir.qubit<1>) -> ()
        // CHECK: quir.barrier %{{.*}}, %{{.*}} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
        quir.barrier %qb1, %qc1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
        // CHECK: quir.call_defcal_gate @defcalGate1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        quir.call_defcal_gate @defcalGate1(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // CHECK: %{{.*}} = quir.call_defcal_measure @defcalMeas1(%{{.*}}) : (!quir.qubit<1>) -> i1
        %ca3 = quir.call_defcal_measure @defcalMeas1(%qa1) : (!quir.qubit<1>) -> i1
        // CHECK: %{{.*}} = "oq3.cast"(%{{.*}}) : (i1) -> !quir.angle<20>
        %ang = "oq3.cast"(%ca3) : (i1) -> !quir.angle<20>
        // CHECK: %{{.*}} = quir.constant #quir.duration<"10ns" : !quir.duration>
        %len1 = quir.constant #quir.duration<"10ns" : !quir.duration>
        // CHECK: %{{.*}} = oq3.declare_stretch : !quir.stretch
        %s1 = "oq3.declare_stretch"() : () -> !quir.stretch
        // CHECK: %{{.*}} = oq3.declare_stretch : !quir.stretch
        %s2 = oq3.declare_stretch : !quir.stretch
        // CHECK: quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
        "quir.delay"(%len1, %qb1) : (!quir.duration, !quir.qubit<1>) -> ()
        // CHECK: quir.delay %{{.*}}, () : !quir.duration, () -> ()
        "quir.delay"(%len1) : (!quir.duration) -> ()
        // CHECK: qcs.delay_cycles() {time = 1000 : i64} : () -> ()
        qcs.delay_cycles () {time = 1000 : i64} : () -> ()
        // CHECK: qcs.delay_cycles(%{{.*}}) {time = 1000 : i64} : (!quir.qubit<1>) -> ()
        qcs.delay_cycles (%qb1) {time = 1000 : i64} : (!quir.qubit<1>) -> ()
        // CHECK: qcs.send %{{.*}} to 1 : i1
        qcs.send %val to 1 : i1
        // CHECK: %{{.*}} = qcs.recv : i1
        %cb3 = qcs.recv : i1
        // CHECK: %{{.*}} = qcs.recv {fromId = [10 : index]} : i1
        %cb4 = qcs.recv {fromId = [10 : index]} : i1
        // CHECK: %{{.*}}:2 = qcs.recv {fromIds = [1 : index, 2 : index]} : i1, i1
        %mResult:2 = qcs.recv {fromIds = [1 : index, 2 : index]} : i1, i1
        // CHECK: qcs.broadcast %{{.*}} : i1
        qcs.broadcast %val : i1
        %ub = arith.constant 10 : index
        // CHECK: quir.call_subroutine @subroutine1(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
        quir.call_subroutine @subroutine1(%qa1, %ang, %ub) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
        // CHECK: qcs.parallel_control_flow
        qcs.parallel_control_flow {
            qcs.parallel_control_flow_end // including just for parsing and verification
        }
        return
    }
}

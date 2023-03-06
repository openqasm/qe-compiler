// RUN: qss-compiler -X=mlir --quir-arg-specialization %s | FileCheck %s

//
// This code is part of Qiskit.
//
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

module {
    quir.declare_variable @cb1 : !quir.cbit<1>
    quir.declare_variable @cb2 : !quir.cbit<1>

    func @gateCall1(%q1 : !quir.qubit<1>, %lambda : !quir.angle<1>) {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle<1>>
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        "quir.call_gate"(%q1) {callee = @proto} : (!quir.qubit<1>) -> ()
        return
    }
    func @gateCall2(%q1 : !quir.qubit<1>, %lambda : !quir.angle) {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle<20>>
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        %cb2 = "quir.measure"(%q1) : (!quir.qubit<1>) -> (i1)
        cond_br %cb2, ^runz, ^dontrunz

    ^runz:
        "quir.call_gate"(%q1) {callee = @Z} : (!quir.qubit<1>) -> ()
        br ^afterz

    ^dontrunz:
        br ^afterz

    ^afterz:
        // if(c1==1) { x q[2]; } // braces optional in this case
        cond_br %cb2, ^runx, ^dontrunx

    // this checks for both specialized and non-specialized intermediate qubit gate calls
    // CHECK: quir.call_gate @X(%{{.*}}) : (!quir.qubit<1>) -> ()
    ^runx:
        "quir.call_gate"(%q1) {callee = @X} : (!quir.qubit<1>) -> ()
        br ^afterx

    ^dontrunx:
        br ^afterx

    ^afterx:
        // post q[2]; // NOP/identity
        // c2 = measure q[2];
        %cb3 = "quir.measure"(%q1) : (!quir.qubit<1>) -> (i1)

        return
    }
    func @gateCall3(%q1 : !quir.qubit<1>, %phi : !quir.angle) {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle<20>>
        %cmpval = quir.constant #quir.angle<0.3 : !quir.angle<20>>
        %farg = "quir.cast"(%phi) : (!quir.angle) -> f64
        %cval = "quir.cast"(%cmpval) : (!quir.angle<20>) -> f64
        %cond = arith.cmpf "ogt", %farg, %cval : f64
        scf.if %cond {
            quir.builtin_U %q1, %zero, %zero, %phi : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        } else {
            quir.builtin_U %q1, %phi, %phi, %phi : !quir.qubit<1>, !quir.angle, !quir.angle, !quir.angle
        }
        return
    }
    func @main () -> i32 {
        %qa1 = quir.declare_qubit { id = 1 : i32 } : !quir.qubit<1>
        %qb1 = quir.declare_qubit { id = 2 : i32 } : !quir.qubit<1>
        %qc1 = quir.declare_qubit { id = 3 : i32 } : !quir.qubit<1>
        quir.reset %qc1 : !quir.qubit<1>
        %cb1 = quir.use_variable @cb1 : !quir.cbit<1>
        %theta = quir.constant #quir.angle<0.1 : !quir.angle<1>>
        // CHECK: quir.call_gate @gateCall1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall1} : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // CHECK: quir.call_gate @"gateCall2_!quir.qubit<1>_!quir.angle<1>"(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall2} : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // CHECK: quir.call_gate @"gateCall2_!quir.qubit<1>_!quir.angle<1>"(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall2} : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1) {callee = @proto} : (!quir.qubit<1>) -> ()
        quir.call_gate @proto(%qb1) : (!quir.qubit<1>) -> ()
        %cb2 = quir.use_variable @cb2 : !quir.cbit<1>
        // CHECK: %{{.*}} = quir.call_kernel @kernel1(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.cbit<1>, !quir.cbit<1>, !quir.cbit<1>) -> !quir.cbit<1>
        %cc1 = quir.call_kernel @kernel1(%cb2, %cb2, %cb2) : (!quir.cbit<1>, !quir.cbit<1>, !quir.cbit<1>) -> !quir.cbit<1>
        // CHECK: quir.call_defcal_gate @defcalGate1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        quir.call_defcal_gate @defcalGate1(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // CHECK %{{.*}} = quir.call_defcal_measure @defcalMeas1(%{{.*}}) : (!quir.qubit<1>) -> i1
        %ca3 = quir.call_defcal_measure @defcalMeas1(%qa1) : (!quir.qubit<1>) -> i1
        quir.call_defcal_gate @defcalGate2(%qb1, %theta) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        quir.call_gate @gateCall3(%qb1, %theta) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        %false = arith.constant false
        scf.if %false {
            %theta2 = quir.constant #quir.angle<0.1 : !quir.angle<3>>
            // CHECK: quir.call_gate @"gateCall3_!quir.qubit<1>_!quir.angle<3>"(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<3>) -> ()
            quir.call_gate @gateCall3(%qb1, %theta2) : (!quir.qubit<1>, !quir.angle<3>) -> ()
        }
        %zero = arith.constant 0 : i32
        return %zero : i32
    }
}

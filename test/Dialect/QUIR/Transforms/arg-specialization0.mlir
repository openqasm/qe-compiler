// RUN: qss-compiler -X=mlir --quir-arg-specialization %s | FileCheck %s
// RUN: qss-compiler -X=mlir %s | FileCheck --check-prefix MLIR %s
//
// This test case validates MLIR with and without argument specialization.

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

    func private @kernel1 (%ca1 : !quir.cbit<1>, %ca2 : !quir.cbit<1>, %ca3 : !quir.cbit<1>) -> !quir.cbit<1>
    func private @kernel2 (memref<?xi1>) -> memref<1xi1>
    func private @proto (%qa1 : !quir.qubit<1>) -> ()
    func @gateCall1(%q1 : !quir.qubit<1>, %lambda : !quir.angle<1>) -> () {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle<1>>
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        "quir.call_gate"(%q1) {callee = @proto} : (!quir.qubit<1>) -> ()
        return
    }
    func @gateCall2(%q1 : !quir.qubit<1>, %lambda : !quir.angle) -> () {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle<20>>
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle
        return
    }
    func @multiQubitGateCall(%qa1 : !quir.qubit<1>, %qb1 : !quir.qubit<1>) -> () {
        quir.builtin_CX %qa1, %qb1 : !quir.qubit<1>, !quir.qubit<1>
        return
    }
    func @defcalGate2(%qa1 : !quir.qubit<1> {quir.qubit_id = 0 : i32}, %theta : !quir.angle {quir.value = 0.5 : f64}) -> () {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle>
        quir.builtin_U %qa1, %zero, %theta, %zero : !quir.qubit<1>, !quir.angle, !quir.angle, !quir.angle
        return
    }
    func @defcalRX(%qa1 : !quir.qubit<1>, %theta : !quir.angle<20>) -> () {
        quir.call_gate @proto1(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<20>) -> ()
        return
    }
    func @defcalRX_q0(%qa1 : !quir.qubit<1> {quir.qubit_id = 0 : i32}, %theta : !quir.angle<20>) -> () attributes {quir.orig_func_name = "defcalRX"} {
        quir.call_gate @proto2(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<20>) -> ()
        return
    }
    func @defcalRX_q0_api2(%qa1 : !quir.qubit<1> {quir.qubit_id = 0 : i32}, %theta : !quir.angle<20> {quir.value = 0.5 : f64}) -> ()
        attributes {quir.orig_func_name = "defcalRX"}
    {
        quir.call_gate @proto3(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<20>) -> ()
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
        // MLIR: quir.call_gate @gateCall1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall1} : (!quir.qubit<1>, !quir.angle<1>) -> ()

        // CHECK: quir.call_gate @"gateCall2_!quir.qubit<1>_!quir.angle<1>"(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // MLIR: quir.call_gate @gateCall2(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall2} : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1, %theta) {callee = @gateCall2} : (!quir.qubit<1>, !quir.angle<1>) -> ()
        "quir.call_gate"(%qb1) {callee = @proto} : (!quir.qubit<1>) -> ()
        quir.call_gate @proto(%qb1) : (!quir.qubit<1>) -> ()
        %cb2 = quir.use_variable @cb2 : !quir.cbit<1>

        // CHECK: %{{.*}} = quir.call_kernel @kernel1(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.cbit<1>, !quir.cbit<1>, !quir.cbit<1>) -> !quir.cbit<1>
        // MLIR: %{{.*}} = quir.call_kernel @kernel1(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.cbit<1>, !quir.cbit<1>, !quir.cbit<1>) -> !quir.cbit<1>
        %cc1 = quir.call_kernel @kernel1(%cb2, %cb2, %cb2) : (!quir.cbit<1>, !quir.cbit<1>, !quir.cbit<1>) -> !quir.cbit<1>

        // CHECK: quir.call_defcal_gate @defcalGate1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // MLIR: quir.call_defcal_gate @defcalGate1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        quir.call_defcal_gate @defcalGate1(%qa1, %theta) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        // CHECK %{{.*}} = quir.call_defcal_measure @defcalMeas1 (%{{.*}}) : (!quir.qubit<1>) -> (i1)
        // MLIR: %{{.*}} = quir.call_defcal_measure @defcalMeas1(%{{.*}}) : (!quir.qubit<1>) -> i1
        %ca3 = quir.call_defcal_measure @defcalMeas1 (%qa1) : (!quir.qubit<1>) -> (i1)
        quir.call_defcal_gate @defcalGate2(%qb1, %theta) : (!quir.qubit<1>, !quir.angle<1>) -> ()
        %zero = arith.constant 0 : i32
        return %zero : i32
    }
}

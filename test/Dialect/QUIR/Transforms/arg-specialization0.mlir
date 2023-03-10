// RUN: qss-compiler -X=mlir --quir-arg-specialization %s | FileCheck %s
// RUN: qss-compiler -X=mlir %s | FileCheck --check-prefix MLIR %s
//
// This test case validates MLIR with and without argument specialization.

module {
    oq3.declare_variable @cb1 : !quir.cbit<1>
    oq3.declare_variable @cb2 : !quir.cbit<1>

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
        %cb1 = oq3.use_variable @cb1 : !quir.cbit<1>
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
        %cb2 = oq3.use_variable @cb2 : !quir.cbit<1>

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

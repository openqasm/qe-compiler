// RUN: qss-compiler -X=mlir --break-reset %s | FileCheck %s
// RUN: qss-compiler -X=mlir --break-reset='numIterations=2 delayCycles=500' %s | FileCheck %s --check-prefix DELAYITER

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


func @main() {
// CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// CHECK: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// DELAYITER: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// DELAYITER: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// DELAYITER: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>

  %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %3 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>

// CHECK-NOT: quir.reset
// DELAYITER-NOT: quir.reset
  quir.reset %1, %2, %3 : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>

// CHECK: [[MEASUREMENT:%.*]]:3 = quir.measure(%0, %1, %2) {quir.noReportRuntime} : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1)
// CHECK: scf.if [[MEASUREMENT]]#0 {
// CHECK:   quir.call_gate @x([[QUBIT0]]) : (!quir.qubit<1>) -> ()
// CHECK: }
// CHECK: scf.if [[MEASUREMENT]]#1 {
// CHECK:   quir.call_gate @x([[QUBIT1]]) : (!quir.qubit<1>) -> ()
// CHECK: }
// CHECK: scf.if [[MEASUREMENT]]#2 {
// CHECK:   quir.call_gate @x([[QUBIT2]]) : (!quir.qubit<1>) -> ()
// CHECK: }


// DELAYITER: [[DURATION:%.*]] = quir.declare_duration {value = "500dt"} : !quir.duration
// DELAYITER: quir.measure
// DELAYITER-COUNT-3: scf.if
// DELAYITER: quir.delay [[DURATION]], ([[QUBIT0]]) : !quir.duration, (!quir.qubit<1>) -> ()
// DELAYITER: quir.delay [[DURATION]], ([[QUBIT1]]) : !quir.duration, (!quir.qubit<1>) -> ()
// DELAYITER: quir.delay [[DURATION]], ([[QUBIT2]]) : !quir.duration, (!quir.qubit<1>) -> ()
// DELAYITER: quir.measure
// DELAYITER-COUNT-3: scf.if

  return
}

// CHECK-NOT: quir.reset
// DELAY-ITER-NOT: quir.reset

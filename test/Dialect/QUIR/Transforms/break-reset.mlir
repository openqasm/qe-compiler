// RUN: qss-compiler -X=mlir --break-reset %s | FileCheck %s
// RUN: qss-compiler -X=mlir --break-reset='delayCycles=500 numIterations=3' %s | FileCheck %s --check-prefix DELAY
// RUN: qss-compiler -X=mlir --break-reset='numIterations=2' %s | FileCheck %s --check-prefix ITER
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

func.func @t1 (%inq : !quir.qubit<1>) {
// CHECK:     %0 = quir.measure(%arg0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
// CHECK:     scf.if %0 {
// CHECK:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// CHECK:     }

// DELAY:     [[DURATION:%.*]] = quir.constant #quir.duration<5.000000e+02> : !quir.duration<dt>
// DELAY-COUNT-2: quir.delay [[DURATION]], ({{.*}}) : !quir.duration<dt>, (!quir.qubit<1>) -> ()

// ITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// ITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// ITER-NOT:   quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()

// DELAYITER:     %dur = quir.constant #quir.duration<5.000000e+02> : !quir.duration<dt>
// DELAYITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// DELAYITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// DELAYITER-NOT:   quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()

  quir.reset %inq : !quir.qubit<1>
  return
}

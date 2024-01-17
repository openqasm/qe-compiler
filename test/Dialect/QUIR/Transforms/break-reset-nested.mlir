// RUN: qss-compiler -X=mlir --break-reset %s | FileCheck %s

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

// CHECK: scf.if %arg1 {
// CHECK-NEXT: scf.if %arg2 {
// CHECK-NEXT:   %{{.*}} = quir.measure(%arg0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
func.func @main (%inq : !quir.qubit<1>, %cond1 : i1, %cond2 : i1) {
  scf.if %cond1 {
    scf.if %cond2 {
      quir.reset %inq : !quir.qubit<1>
    }
  }
  return
}

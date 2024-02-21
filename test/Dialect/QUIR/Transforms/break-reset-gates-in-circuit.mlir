// RUN: qss-compiler -X=mlir --break-reset='putCallGatesAndMeasuresIntoCircuit=true' %s | FileCheck %s --check-prefix CHECK

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

module {
// CHECK: quir.circuit @reset_measure_0(%arg0: !quir.qubit<1>) -> i1 {
// CHECK: %0 = quir.measure(%arg0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
// CHECK: quir.return %0 : i1

// CHECK: quir.circuit @reset_x_1(%arg0: !quir.qubit<1>) {
// CHECK: quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// CHECK: quir.return
func.func @main (%inq : !quir.qubit<1>) {

// CHECK: %0 = quir.call_circuit @reset_measure_0(%arg0) : (!quir.qubit<1>) -> i1
// CHECK-NOT: %0 = quir.measure(%arg0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
// CHECK: scf.if %0 {
// CHECK: quir.call_circuit @reset_x_1(%arg0) : (!quir.qubit<1>) -> ()
// CHECK-NOT: quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()

  quir.reset %inq : !quir.qubit<1>
  return
}}

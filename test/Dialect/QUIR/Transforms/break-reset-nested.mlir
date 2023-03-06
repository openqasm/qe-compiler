// RUN: qss-compiler -X=mlir --break-reset %s | FileCheck %s

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

// CHECK: scf.if %arg1 {
// CHECK-NEXT: scf.if %arg2 {
// CHECK-NEXT:   %{{.*}} = quir.measure(%arg0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
func @main (%inq : !quir.qubit<1>, %cond1 : i1, %cond2 : i1) {
  scf.if %cond1 {
    scf.if %cond2 {
      quir.reset %inq : !quir.qubit<1>
    }
  }
  return
}

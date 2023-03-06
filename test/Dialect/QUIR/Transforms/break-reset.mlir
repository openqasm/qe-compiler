// RUN: qss-compiler -X=mlir --break-reset %s | FileCheck %s
// RUN: qss-compiler -X=mlir --break-reset='delayCycles=500 numIterations=3' %s | FileCheck %s --check-prefix DELAY
// RUN: qss-compiler -X=mlir --break-reset='numIterations=2' %s | FileCheck %s --check-prefix ITER
// RUN: qss-compiler -X=mlir --break-reset='numIterations=2 delayCycles=500' %s | FileCheck %s --check-prefix DELAYITER

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

func @t1 (%inq : !quir.qubit<1>) {
// CHECK:     %0 = quir.measure(%arg0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
// CHECK:     scf.if %0 {
// CHECK:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// CHECK:     }

// DELAY:     [[DURATION:%.*]] = quir.declare_duration {value = "500dt"} : !quir.duration
// DELAY-COUNT-2: quir.delay [[DURATION]], ({{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()

// ITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// ITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// ITER-NOT:   quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()

// DELAYITER:     %0 = quir.declare_duration {value = "500dt"} : !quir.duration
// DELAYITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// DELAYITER:       quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
// DELAYITER-NOT:   quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()

  quir.reset %inq : !quir.qubit<1>
  return
}

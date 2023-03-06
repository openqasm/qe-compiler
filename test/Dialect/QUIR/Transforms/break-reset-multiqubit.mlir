// RUN: qss-compiler -X=mlir --break-reset %s | FileCheck %s
// RUN: qss-compiler -X=mlir --break-reset='numIterations=2 delayCycles=500' %s | FileCheck %s --check-prefix DELAYITER

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

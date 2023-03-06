// RUN: qss-compiler --canonicalize --reorder-measures %s | FileCheck %s

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

// CHECK: module
module {
  quir.declare_variable @results : !quir.cbit<1>
  func @cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    %5 = "quir.cast"(%4) : (i1) -> !quir.cbit<1>
    quir.assign_variable @results : !quir.cbit<1> = %5
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    %6 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    %7 = "quir.cast"(%6) : (i1) -> !quir.cbit<1>
    quir.assign_variable @results : !quir.cbit<1> = %7
    return %c0_i32 : i32
  }
}

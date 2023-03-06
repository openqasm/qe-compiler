// RUN: qss-compiler -X=mlir --canonicalize %s | FileCheck %s --implicit-check-not assign_cbit_bit
//
// This test case validates that the single-bit cbit assignments
// are simplified to variable assignments by
// AssignSingleCbitToAssignVariablePattern.

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
  quir.declare_variable @a : !quir.cbit<1>
  quir.declare_variable @b : !quir.cbit<1>

  func @main() -> i32 {
    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

    // CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0
    // CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1

    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    quir.assign_cbit_bit @a<1> [0] : i1 = %5

    // CHECK: [[MEAS0:%.*]] = quir.measure([[QUBIT0]])
    // CHECK: [[CAST0:%.*]] = "quir.cast"([[MEAS0]]) : (i1) -> !quir.cbit<1>
    // CHECK: quir.assign_variable @a : !quir.cbit<1> = [[CAST0]]

    %6 = quir.measure(%2) : (!quir.qubit<1>) -> i1
    quir.assign_cbit_bit @b<1> [0] : i1 = %6

    // CHECK: [[MEAS1:%.*]] = quir.measure([[QUBIT1]])
    // CHECK: [[CAST1:%.*]] = "quir.cast"([[MEAS1]]) : (i1) -> !quir.cbit<1>
    // CHECK: quir.assign_variable @b : !quir.cbit<1> = [[CAST1]]

    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

// RUN: qss-compiler -X=mlir --canonicalize %s | FileCheck %s --implicit-check-not extui --implicit-check-not cmpi --implicit-check-not cast
//
// This test case validates that comparisons between zero-extended i1 and
// constant 1 for equality are simplified.

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

// CHECK: module
module {
  func @extract(%in : !quir.cbit<2>) -> i1 {
    // CHECK: [[BIT:%.]] = quir.cbit_extractbit
    %1 = quir.cbit_extractbit(%in : !quir.cbit<2>) [0] : i1
    %c1_i32 = arith.constant 1 : i32
    %2 = "quir.cast"(%1): (i1) -> i32
    %3 = arith.cmpi eq, %2, %c1_i32 : i32
    // CHECK: return [[BIT]] : i1
    return %3 : i1
  }

  func @main() -> i1 {
    %c1_i32 = arith.constant 1 : i32

    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // CHECK: [[MEASURE:%.*]] = quir.measure
    %2 = quir.measure(%1) : (!quir.qubit<1>) -> i1

    %3 = arith.extui %2 : i1 to i32
    %4 = arith.cmpi eq, %3, %c1_i32 : i32

    // CHECK: return [[MEASURE]]
    return %4 : i1
  }
}

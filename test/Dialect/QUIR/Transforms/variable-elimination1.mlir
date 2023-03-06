// RUN: qss-compiler -X=mlir --canonicalize --quir-eliminate-variables %s | FileCheck %s --implicit-check-not load --implicit-check-not alloc --implicit-check-not store

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

// This test verifies store-forwarding and the removal of invisible stores. All
// variable loads must be replaced by forwarded stored values. Then, any
// remaining stores are invisible as the variables have no lifetime beyond this
// program and are to be removed, together with the allocation of variables.
//
// CHECK: module
module {
  quir.declare_variable @a : !quir.cbit<1>
  quir.declare_variable @b : !quir.cbit<1>
  func @x(%arg0: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    // CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32}
    // CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32}
    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

    %false = arith.constant false
    %3 = "quir.cast"(%false) : (i1) -> !quir.cbit<1>
    quir.assign_variable @a : !quir.cbit<1> = %3
    %false_0 = arith.constant false
    %4 = "quir.cast"(%false_0) : (i1) -> !quir.cbit<1>
    quir.assign_variable @b : !quir.cbit<1> = %4

    // CHECK: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]])
    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    quir.assign_cbit_bit @a<1> [0] : i1 = %5

    // CHECK: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]])
    %6 = quir.measure(%2) : (!quir.qubit<1>) -> i1
    quir.assign_cbit_bit @b<1> [0] : i1 = %6

    %7 = quir.use_variable @a : !quir.cbit<1>
    %c1_i32 = arith.constant 1 : i32
    // measurement value has been forwarded, there is no load
    // CHECK: arith.extui [[MEASURE0]]
    %8 = "quir.cast"(%7) : (!quir.cbit<1>) -> i32
    %9 = arith.cmpi eq, %8, %c1_i32 : i32

    scf.if %9 {
      quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
      %cst = constant unit
    }
    %10 = quir.use_variable @b : !quir.cbit<1>
    %c1_i32_1 = arith.constant 1 : i32
    %11 = "quir.cast"(%10) : (!quir.cbit<1>) -> i32

    // measurement value has been forwarded, there is no load
    // CHECK: arith.extui [[MEASURE1]]
    %12 = arith.cmpi eq, %11, %c1_i32_1 : i32
    scf.if %12 {
      quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
      %cst = constant unit
    }

    %c0_i32 = arith.constant 0 : i32
    return %8 : i32 // %c0_i32 : i32
  }
}

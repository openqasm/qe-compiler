// RUN: qss-compiler -X=mlir --quir-eliminate-loads %s | FileCheck %s --implicit-check-not 'quir.use_variable @a'
// RUN: qss-compiler -X=mlir --quir-eliminate-loads --remove-unused-variables %s | FileCheck %s --check-prefix REMOVE-UNUSED
//
// This test case serves to validate the behavior of the load elimination pass.

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

module {

  // CHECK: quir.declare_variable @a : i32
  // CHECK: quir.declare_variable @b : i32
  quir.declare_variable @a : i32
  quir.declare_variable @b : i32

  // REMOVE-UNUSED-NOT: quir.declare_variable @a

  func @main() -> i32 {
    %c1 = arith.constant 1 : index

    // CHECK: [[CONST17_I32:%.*]] = arith.constant 17 : i32
    %c17_i32 = arith.constant 17 : i32
    quir.assign_variable @a : i32 = %c17_i32

    // REMOVE-UNUSED-NOT: quir.assign_variable @a

    %c1_i32_0 = arith.constant 1 : i32
    quir.assign_variable @b : i32 = %c1_i32_0

    // The load elimination pass should forward-propagate the initializer to the
    // assignment of b.
    // CHECK: quir.assign_variable @b : i32 = [[CONST17_I32]]
    // The variable a should never be read.
    // REMOVE-UNUSED-NOT: quir.use_variable @a
    %1 = quir.use_variable @a : i32
    quir.assign_variable @b : i32 = %1

    %2 = quir.use_variable @b : i32
    return %2 : i32
  }
}

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

  // CHECK: quir.declare_variable {input} @a : !quir.angle<64>
  // CHECK: quir.declare_variable @b : !quir.angle<64>
  quir.declare_variable {input} @a : !quir.angle<64>
  quir.declare_variable @b : !quir.angle<64>

  // REMOVE-UNUSED-NOT: quir.declare_variable {input} @a

  func @main() -> !quir.angle<64> {

    // CHECK: [[CONST314_ANGLE:%.*]] = quir.constant {quir.inputParameter = "a"} #quir.angle<3.140000e+00 : !quir.angle<64>>
    // REMOVE-UNUSED: [[CONST314_ANGLE:%.*]] = quir.constant #quir.angle<3.140000e+00 : !quir.angle<64>>
    %angle = quir.constant #quir.angle<3.140000e+00 : !quir.angle<64>>
    %angle2 = quir.constant #quir.angle<3.140000e+00 : !quir.angle<64>>

    // REMOVE-UNUSED-NOT: quir.assign_variable @a
    quir.assign_variable @a : !quir.angle<64> = %angle

    %angle_0 = quir.constant #quir.angle<1.000000e+00 : !quir.angle<64>>
    quir.assign_variable @b : !quir.angle<64> = %angle_0

    // The load elimination pass should forward-propagate the initializer to the
    // assignment of b.
    // CHECK: quir.assign_variable @b : !quir.angle<64> = [[CONST314_ANGLE]]
    // The variable a should never be read.
    // REMOVE-UNUSED-NOT: quir.use_variable @a
    %1 = quir.use_variable @a : !quir.angle<64>
    quir.assign_variable @b : !quir.angle<64> = %1

    %2 = quir.use_variable @b : !quir.angle<64>
    return %2 : !quir.angle<64>
  }
}

// RUN: qss-compiler -X=mlir --convert-quir-angles %s | FileCheck %s

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

module  {
  func @rz(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) {
    return
  }
  func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<32>>
    quir.call_gate @rz(%0, %1) : (!quir.qubit<1>, !quir.angle<32>) -> ()
    qcs.finalize
    return %c0_i32 : i32
  }
}

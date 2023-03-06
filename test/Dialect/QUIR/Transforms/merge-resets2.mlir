// RUN: qss-compiler -X=mlir --merge-resets-lexicographic -emit=mlir %s | FileCheck %s
//
// This test case checks that the MergeResetPass merges all parallelizable
// resets (and no more than that) and removes the resets that have been merged
// into other operations.

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
  func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      %0 = quir.declare_duration {value = "1ms"} : !quir.duration
      quir.delay %0, () : !quir.duration, () -> ()

      // qubit $0;
      // qubit $1;
      %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      // CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      // CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

      // Expect that parallelizable resets get merged
      // reset $0;
      // reset $1;
      quir.reset %1 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT0]] : !quir.qubit<1>
      quir.reset %2 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT1]] : !quir.qubit<1>
      // CHECK: quir.reset [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>

    }
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

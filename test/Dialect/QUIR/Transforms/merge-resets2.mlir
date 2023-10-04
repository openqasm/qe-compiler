// RUN: qss-compiler -X=mlir --merge-resets-lexicographic -emit=mlir %s | FileCheck %s
//
// This test case checks that the MergeResetPass merges all parallelizable
// resets (and no more than that) and removes the resets that have been merged
// into other operations.

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

module  {
  func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      %0 = quir.constant #quir.duration<1 : !quir.duration<ms>>
      quir.delay %0, () : !quir.duration<ms>, () -> ()

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

// RUN: qss-compiler -X=mlir --convert-quir-angles %s | FileCheck %s

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
  func.func @rz(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) {
    return
  }
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<32>>
    quir.call_gate @rz(%0, %1) : (!quir.qubit<1>, !quir.angle<32>) -> ()
    qcs.finalize
    return %c0_i32 : i32
  }
}

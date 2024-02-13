// RUN: qss-compiler %s --config %TEST_CFG --target mock --canonicalize --mock-quir-to-std --emit=qem --plaintext-payload  | FileCheck %s

// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// CHECK: define i32 @main() {
// CHECK:  ret i32 0
// CHECK: }
module @controller attributes {quir.nodeId = 1000 : ui32, quir.nodeType = "controller"}  {
  func.func @main() -> i32 attributes {quir.classicalOnly = false} {
    %0 = quir.constant #quir.duration<1000.0> : !quir.duration<dt>
    %1 = qcs.recv {fromId = 0 : index} : i1
    qcs.broadcast %1 : i1
    scf.if %1 {
    } {quir.classicalOnly = false}
    %2 = qcs.recv {fromId = 0 : index} : i1
    qcs.broadcast %2 : i1
    scf.if %2 {
    } {quir.classicalOnly = false}
    %3 = qcs.recv {fromId = 0 : index} : i1
    qcs.broadcast %3 : i1
    scf.if %3 {
    } {quir.classicalOnly = false}
    %4 = llvm.mlir.constant(1.0) : f32
    %5 = llvm.mlir.constant(2.0) : f32
    %6 = "llvm.intr.pow"(%4, %5) : (f32, f32) -> f32
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

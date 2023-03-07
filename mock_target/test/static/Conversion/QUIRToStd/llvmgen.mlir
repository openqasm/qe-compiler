// RUN: qss-compiler %s --config %TEST_CFG --target mock --canonicalize --mock-quir-to-std --emit=qem --plaintext-payload  | FileCheck %s

// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
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

// CHECK: define i32 @main() !dbg !3 {
// CHECK:  ret i32 0, !dbg !7
// CHECK: }
module @controller attributes {quir.nodeId = 1000 : i32, quir.nodeType = "controller"}  {
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %0 = quir.declare_duration {value = "1000dt"} : !quir.duration
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
    %4 = llvm.mlir.constant(1.0) : i32
    %5 = llvm.mlir.constant(2.0) : i32
    %6 = "llvm.intr.pow"(%4, %5) : (i32, i32) -> i32
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

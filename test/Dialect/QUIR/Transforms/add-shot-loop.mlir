// RUN: qss-compiler -X=mlir --add-shot-loop %s | FileCheck %s

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

func.func @main() {
  qcs.init
  // CHECK: scf.for
  // CHECK: qcs.shot_init
  // CHECK: qcs.shot_loop
  qcs.finalize
  return
}

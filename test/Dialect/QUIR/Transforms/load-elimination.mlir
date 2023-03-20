// RUN: qss-compiler -X=mlir --quir-eliminate-loads %s | FileCheck %s --implicit-check-not 'quir.use_variable @a'
// RUN: qss-compiler -X=mlir --quir-eliminate-loads --remove-unused-variables %s | FileCheck %s --check-prefix REMOVE-UNUSED
//
// This test case serves to validate the behavior of the load elimination pass.

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

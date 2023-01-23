// RUN: qss-compiler -X=mlir --quir-eliminate-loads %s | FileCheck %s --implicit-check-not 'quir.use_variable @a'
// RUN: qss-compiler -X=mlir --quir-eliminate-loads --remove-unused-variables %s | FileCheck %s --check-prefix REMOVE-UNUSED
//
// This test case serves to validate the behavior of the load elimination pass.

module {

  // CHECK: oq3.declare_variable @a : i32
  // CHECK: oq3.declare_variable @b : i32
  oq3.declare_variable @a : i32
  oq3.declare_variable @b : i32

  // REMOVE-UNUSED-NOT: oq3.declare_variable @a

  func @main() -> i32 {
    %c1 = arith.constant 1 : index

    // CHECK: [[CONST17_I32:%.*]] = arith.constant 17 : i32
    %c17_i32 = arith.constant 17 : i32
    oq3.assign_variable @a : i32 = %c17_i32

    // REMOVE-UNUSED-NOT: oq3.assign_variable @a

    %c1_i32_0 = arith.constant 1 : i32
    oq3.assign_variable @b : i32 = %c1_i32_0

    // The load elimination pass should forward-propagate the initializer to the
    // assignment of b.
    // CHECK: oq3.assign_variable @b : i32 = [[CONST17_I32]]
    // The variable a should never be read.
    // REMOVE-UNUSED-NOT: quir.use_variable @a
    %1 = quir.use_variable @a : i32
    oq3.assign_variable @b : i32 = %1

    %2 = quir.use_variable @b : i32
    return %2 : i32
  }
}

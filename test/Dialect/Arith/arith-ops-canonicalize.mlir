// RUN: qss-compiler %s  --emit=mlir --canonicalize | FileCheck %s

// Check that constant folding is applied by canonicalization

module {
   oq3.declare_variable @out0 : i32
   oq3.declare_variable @out1 : i32
   oq3.declare_variable @out2 : i32
   oq3.declare_variable @out3 : i32
   oq3.declare_variable @out4 : i32
   oq3.declare_variable @out5 : f32
   oq3.declare_variable @out6 : f32
   oq3.declare_variable @out7 : f32
   oq3.declare_variable @out8 : f32
   oq3.declare_variable @out9 : f32


  func.func @main() -> i32 {
    qcs.init
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      qcs.shot_init {qcs.num_shots = 1000 : i32}
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %cst = arith.constant 1.000000e+00 : f32
      %cst_1 = arith.constant 2.000000e+00 : f32
      %cst_2 = arith.constant 0.000000e+00 : f32

      // CHECK: %[[cst:.*]] = arith.constant 5.000000e-01 : f32
      // CHECK: %[[cst_0:.*]] = arith.constant -1.000000e+00 : f32
      // CHECK: %[[cst_1:.*]] = arith.constant 3.000000e+00 : f32
      // CHECK: %[[c_1_i32:.*]] = arith.constant -1 : i32
      // CHECK: %[[c3_i32:.*]] = arith.constant 3 : i32
      // CHECK: %[[cst_2:.*]] = arith.constant 2.000000e+00 : f32
      // CHECK: %[[cst_3:.*]] = arith.constant 1.000000e+00 : f32
      // CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      // CHECK: %[[c2_i32:.*]] = arith.constant 2 : i32
      // CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32

      %0 = arith.addi %c1_i32, %c2_i32 : i32
      oq3.variable_assign @out0 : i32 = %0
      // CHECK: oq3.variable_assign @out0 : i32 = %[[c3_i32]]

      %1 = arith.subi %c1_i32, %c2_i32 : i32
      oq3.variable_assign @out1 : i32 = %1
      // CHECK: oq3.variable_assign @out1 : i32 = %[[c_1_i32]]

      %2 = arith.muli %c1_i32, %c2_i32 : i32
      oq3.variable_assign @out2 : i32 = %2
      // CHECK: oq3.variable_assign @out2 : i32 = %[[c2_i32]]

      %3 = arith.divsi %c1_i32, %c2_i32 : i32
      oq3.variable_assign @out3 : i32 = %3
      // CHECK: oq3.variable_assign @out3 : i32 = %[[c0_i32]]

      %4 = arith.remsi %c1_i32, %c2_i32 : i32
      oq3.variable_assign @out4 : i32 = %4
      // CHECK: oq3.variable_assign @out4 : i32 = %[[c1_i32]]

      %5 = arith.addf %cst, %cst_1 : f32
      oq3.variable_assign @out5 : f32 = %5
      // CHECK: oq3.variable_assign @out5 : f32 = %[[cst_1]]

      %6 = arith.subf %cst, %cst_1 : f32
      oq3.variable_assign @out6 : f32 = %6
      // CHECK: oq3.variable_assign @out6 : f32 = %[[cst_0]]

      %7 = arith.mulf %cst, %cst_1 : f32
      oq3.variable_assign @out7 : f32 = %7
      // CHECK: oq3.variable_assign @out7 : f32 = %[[cst_2]]

      %8 = arith.divf %cst, %cst_1 : f32
      oq3.variable_assign @out8 : f32 = %8
      // CHECK: oq3.variable_assign @out8 : f32 = %[[cst]]

      %9 = arith.remf %cst, %cst_1 : f32
      oq3.variable_assign @out9 : f32 = %9
      // CHECK: oq3.variable_assign @out9 : f32 = %[[cst_3]]

    } {qcs.shot_loop}
    qcs.finalize
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

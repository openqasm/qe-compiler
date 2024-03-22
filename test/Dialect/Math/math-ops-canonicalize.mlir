// RUN: qss-compiler %s  --emit=mlir --canonicalize | FileCheck %s

// Check that constant folding is applied by canonicalization

module {
   oq3.declare_variable @out0 : i32
   oq3.declare_variable @out1 : f32
   oq3.declare_variable @out2 : f32
   oq3.declare_variable @out3 : f32
   oq3.declare_variable @out4 : f32
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


      // CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
      // CHECK: %[[cst_0:.*]] = arith.constant 2.71828175 : f32
      // CHECK: %[[cst_1:.*]] = arith.constant 0.785398185 : f32
      // CHECK: %[[cst_2:.*]] = arith.constant 1.55740774 : f32
      // CHECK: %[[cst_3:.*]] = arith.constant {{0.841470957|8.414710e-01}} : f32
      // CHECK: %[[cst_4:.*]] = arith.constant 0.540302277 : f32
      // CHECK: %[[cst_5:.*]] = arith.constant 2.000000e+00 : f32
      // CHECK: %[[cst_6:.*]] = arith.constant 1.000000e+00 : f32
      // CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32

      %0 = math.ipowi %c1_i32, %c2_i32 : i32
      oq3.variable_assign @out0 : i32 = %0
      // CHECK: oq3.variable_assign @out0 : i32 = %[[c1_i32]]

      %1 = math.powf %cst, %cst_1 : f32
      oq3.variable_assign @out1 : f32 = %1
      // CHECK: oq3.variable_assign @out1 : f32 = %[[cst_6]]

      %2 = math.fpowi %cst_1, %c1_i32 : f32, i32
      oq3.variable_assign @out2 : f32 = %2

      // CHECK: %0 = math.fpowi %cst_5, %c1_i32 : f32, i32
      // CHECK: oq3.variable_assign @out2 : f32 = %0

      %3 = math.cos %cst : f32
      oq3.variable_assign @out3 : f32 = %3
      // CHECK: oq3.variable_assign @out3 : f32 = %[[cst_4]]

      %4 = math.sin %cst : f32
      oq3.variable_assign @out4 : f32 = %4
      // CHECK: oq3.variable_assign @out4 : f32 = %[[cst_3]]

      %5 = math.tan %cst : f32
      oq3.variable_assign @out5 : f32 = %5
      // CHECK: oq3.variable_assign @out5 : f32 = %[[cst_2]]

      %6 = math.atan %cst : f32
      oq3.variable_assign @out6 : f32 = %6
      // CHECK: oq3.variable_assign @out6 : f32 = %[[cst_1]]

      %7 = math.exp %cst : f32
      oq3.variable_assign @out7 : f32 = %7
      // CHECK: oq3.variable_assign @out7 : f32 = %[[cst_0]]

      %8 = math.log %cst : f32
      oq3.variable_assign @out8 : f32 = %8
      // CHECK: oq3.variable_assign @out8 : f32 = %[[cst]]

      %9 = math.sqrt %cst : f32
      oq3.variable_assign @out9 : f32 = %9
      // CHECK: oq3.variable_assign @out9 : f32 = %[[cst_6]]

    } {qcs.shot_loop}
    qcs.finalize
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

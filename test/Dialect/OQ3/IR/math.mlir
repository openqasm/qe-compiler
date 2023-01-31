// RUN: qss-compiler -X=mlir %s | FileCheck %s
module {
    oq3.variable_decl @cb1 : !quir.cbit<10>
    %cb1 = oq3.use_variable @cb1 : !quir.cbit<10>

    // CHECK: %{{.*}} = oq3.cbit_not %{{.*}} : !quir.cbit<10>
    %cb2 = oq3.cbit_not %cb1 : !quir.cbit<10>
    %const2 = arith.constant 2 : i32
    // CHECK:  oq3.cbit_rotl %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %cb3 = oq3.cbit_rotl %cb2, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    // CHECK: %{{.*}} = oq3.cbit_rotr %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %cb4 = oq3.cbit_rotr %cb2, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    // CHECK: %{{.*}} = oq3.cbit_popcount %{{.*}} : (!quir.cbit<10>) -> i32
    %count = oq3.cbit_popcount %cb4 : (!quir.cbit<10>) -> i32
    // CHECK: %{{.*}} = oq3.cbit_and %{{.*}}, %{{.*}} : !quir.cbit<10>
    %and_res = oq3.cbit_and %cb3, %cb4 : !quir.cbit<10>
    // CHECK: %{{.*}} = oq3.cbit_or %{{.*}}, %{{.*}} : !quir.cbit<10>
    %or_res = oq3.cbit_or %cb3, %cb4 : !quir.cbit<10>
    // CHECK: %{{.*}} = oq3.cbit_xor %{{.*}}, %{{.*}} : !quir.cbit<10>
    %xor_res = oq3.cbit_xor %cb3, %cb4 : !quir.cbit<10>
    // CHECK:  oq3.cbit_rshift %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %rshift_res = oq3.cbit_rshift %xor_res, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    // CHECK:  oq3.cbit_lshift %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %lshift_res = oq3.cbit_lshift %xor_res, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>

    // CHECK: %{{.*}} = quir.constant #quir.angle<1.000000e-01 : !quir.angle<20>>
    %theta = quir.constant #quir.angle<0.1 : !quir.angle<20>>
    %phi = quir.constant #quir.angle<0.2 : !quir.angle<20>>
    // CHECK: %{{.*}} = oq3.angle_add %{{.*}}, %{{.*}} : !quir.angle<20>
    %add_res = oq3.angle_add %theta, %phi : !quir.angle<20>
    // CHECK: %{{.*}} = oq3.angle_sub %{{.*}}, %{{.*}} : !quir.angle<20>
    %sub_res = oq3.angle_sub %theta, %phi : !quir.angle<20>
    // CHECK: %{{.*}} = oq3.angle_mul %{{.*}}, %{{.*}} : !quir.angle<20>
    %mul_res = oq3.angle_mul %theta, %phi : !quir.angle<20>
    // CHECK: %{{.*}} = oq3.angle_div %{{.*}}, %{{.*}} : !quir.angle<20>
    %div_res = oq3.angle_div %theta, %phi : !quir.angle<20>

    // CHECK: %{{.*}} = oq3.declare_duration {value = "10ns"} : !quir.duration
    %l1 = oq3.declare_duration {value = "10ns"} : !quir.duration
    %l2 = oq3.declare_duration {value = "100ns"} : !quir.duration
    // CHECK: %{{.*}} = oq3.duration_add %{{.*}}, %{{.*}} : !quir.duration
    %ladd_res = oq3.duration_add %l1, %l2 : !quir.duration
    // CHECK: %{{.*}} = oq3.duration_sub %{{.*}}, %{{.*}} : !quir.duration
    %lsub_res = oq3.duration_sub %l1, %l2 : !quir.duration
    // CHECK: %{{.*}} = oq3.duration_mul %{{.*}}, %{{.*}} : !quir.duration
    %lmul_res = oq3.duration_mul %l1, %l2 : !quir.duration
}

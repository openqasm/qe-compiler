// RUN: qss-compiler -X=mlir %s | FileCheck %s

// (C) Copyright IBM 2023.
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

module {
    quir.declare_variable @cb1 : !quir.cbit<10>
    %cb1 = quir.use_variable @cb1 : !quir.cbit<10>

    // CHECK: %{{.*}} = quir.cbit_not %{{.*}} : !quir.cbit<10>
    %cb2 = quir.cbit_not %cb1 : !quir.cbit<10>
    %const2 = arith.constant 2 : i32
    // CHECK:  quir.cbit_rotl %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %cb3 = quir.cbit_rotl %cb2, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    // CHECK: %{{.*}} = quir.cbit_rotr %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %cb4 = quir.cbit_rotr %cb2, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    // CHECK: %{{.*}} = quir.cbit_popcount %{{.*}} : (!quir.cbit<10>) -> i32
    %count = quir.cbit_popcount %cb4 : (!quir.cbit<10>) -> i32
    // CHECK: %{{.*}} = quir.cbit_and %{{.*}}, %{{.*}} : !quir.cbit<10>
    %and_res = quir.cbit_and %cb3, %cb4 : !quir.cbit<10>
    // CHECK: %{{.*}} = quir.cbit_or %{{.*}}, %{{.*}} : !quir.cbit<10>
    %or_res = quir.cbit_or %cb3, %cb4 : !quir.cbit<10>
    // CHECK: %{{.*}} = quir.cbit_xor %{{.*}}, %{{.*}} : !quir.cbit<10>
    %xor_res = quir.cbit_xor %cb3, %cb4 : !quir.cbit<10>
    // CHECK:  quir.cbit_rshift %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %rshift_res = quir.cbit_rshift %xor_res, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    // CHECK:  quir.cbit_lshift %{{.*}}, %{{.*}} : (!quir.cbit<10>, i32) -> !quir.cbit<10>
    %lshift_res = quir.cbit_lshift %xor_res, %const2 : (!quir.cbit<10>, i32) -> !quir.cbit<10>

    // CHECK: %{{.*}} = quir.constant #quir.angle<1.000000e-01 : !quir.angle<20>>
    %theta = quir.constant #quir.angle<0.1 : !quir.angle<20>>
    %phi = quir.constant #quir.angle<0.2 : !quir.angle<20>>
    // CHECK: %{{.*}} = quir.angle_add %{{.*}}, %{{.*}} : !quir.angle<20>
    %add_res = quir.angle_add %theta, %phi : !quir.angle<20>
    // CHECK: %{{.*}} = quir.angle_sub %{{.*}}, %{{.*}} : !quir.angle<20>
    %sub_res = quir.angle_sub %theta, %phi : !quir.angle<20>
    // CHECK: %{{.*}} = quir.angle_mul %{{.*}}, %{{.*}} : !quir.angle<20>
    %mul_res = quir.angle_mul %theta, %phi : !quir.angle<20>
    // CHECK: %{{.*}} = quir.angle_div %{{.*}}, %{{.*}} : !quir.angle<20>
    %div_res = quir.angle_div %theta, %phi : !quir.angle<20>

    // CHECK: %{{.*}} = quir.declare_duration {value = "10ns"} : !quir.duration
    %l1 = quir.declare_duration {value = "10ns"} : !quir.duration
    %l2 = quir.declare_duration {value = "100ns"} : !quir.duration
    // CHECK: %{{.*}} = quir.duration_add %{{.*}}, %{{.*}} : !quir.duration
    %ladd_res = quir.duration_add %l1, %l2 : !quir.duration
    // CHECK: %{{.*}} = quir.duration_sub %{{.*}}, %{{.*}} : !quir.duration
    %lsub_res = quir.duration_sub %l1, %l2 : !quir.duration
    // CHECK: %{{.*}} = quir.duration_mul %{{.*}}, %{{.*}} : !quir.duration
    %lmul_res = quir.duration_mul %l1, %l2 : !quir.duration
}

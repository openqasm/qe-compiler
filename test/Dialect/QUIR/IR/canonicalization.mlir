// RUN: qss-compiler -X=mlir --canonicalize %s | FileCheck %s

//
// This code is part of Qiskit.
//
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

// CHECK: func @t1
func @t1 (%arg1 : !quir.cbit<1>) -> (!quir.cbit<1>) {
// CHECK: quir.cbit_not %{{.*}} : !quir.cbit<1>
// CHECK-NOT: quir.cbit_not %{{.*}} : !quir.cbit<1>
    %c2 = quir.cbit_not %arg1 : !quir.cbit<1>
    %c3 = quir.cbit_not %c2 : !quir.cbit<1>
    %c4 = quir.cbit_not %c3 : !quir.cbit<1>
    return %c4 : !quir.cbit<1>
}

// CHECK: func @t2
func @t2 (%arg1 : !quir.cbit<1>) -> (!quir.cbit<1>) {
// CHECK-NOT: quir.cbit_not %{{.*}} : !quir.cbit<1>
// CHECK-NOT: quir.cbit_not %{{.*}} : !quir.cbit<1>
    %c2 = quir.cbit_not %arg1 : !quir.cbit<1>
    %c3 = quir.cbit_not %c2 : !quir.cbit<1>
    return %c3 : !quir.cbit<1>
}

// CHECK: func @t7
func @t7 (%arg0 : i32, %arg1 : f32) -> (i32, f32) {
    %r1 = "quir.cast"(%arg0) : (i32) -> i32
    %r2 = "quir.cast"(%arg1) : (f32) -> f32
    // CHECK: return %arg0, %arg1 : i32, f32
    return %r1, %r2 : i32, f32
}

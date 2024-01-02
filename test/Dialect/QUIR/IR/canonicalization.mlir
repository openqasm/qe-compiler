// RUN: qss-compiler -X=mlir --canonicalize %s | FileCheck %s

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

// CHECK: func.func @t1
func.func @t1 (%arg1 : !quir.cbit<1>) -> (!quir.cbit<1>) {
// CHECK: oq3.cbit_not %{{.*}} : !quir.cbit<1>
// CHECK-NOT: oq3.cbit_not %{{.*}} : !quir.cbit<1>
    %c2 = oq3.cbit_not %arg1 : !quir.cbit<1>
    %c3 = oq3.cbit_not %c2 : !quir.cbit<1>
    %c4 = oq3.cbit_not %c3 : !quir.cbit<1>
    return %c4 : !quir.cbit<1>
}

// CHECK: func.func @t2
func.func @t2 (%arg1 : !quir.cbit<1>) -> (!quir.cbit<1>) {
// CHECK-NOT: oq3.cbit_not %{{.*}} : !quir.cbit<1>
// CHECK-NOT: oq3.cbit_not %{{.*}} : !quir.cbit<1>
    %c2 = oq3.cbit_not %arg1 : !quir.cbit<1>
    %c3 = oq3.cbit_not %c2 : !quir.cbit<1>
    return %c3 : !quir.cbit<1>
}

// CHECK: func.func @t7
func.func @t7 (%arg0 : i32, %arg1 : f32) -> (i32, f32) {
    %r1 = "oq3.cast"(%arg0) : (i32) -> i32
    %r2 = "oq3.cast"(%arg1) : (f32) -> f32
    // CHECK: return %arg0, %arg1 : i32, f32
    return %r1, %r2 : i32, f32
}

// RUN: qss-compiler -X=mlir --canonicalize %s | FileCheck %s

// CHECK: func @t1
func @t1 (%arg1 : !quir.cbit<1>) -> (!quir.cbit<1>) {
// CHECK: oq3.cbit_not %{{.*}} : !quir.cbit<1>
// CHECK-NOT: oq3.cbit_not %{{.*}} : !quir.cbit<1>
    %c2 = oq3.cbit_not %arg1 : !quir.cbit<1>
    %c3 = oq3.cbit_not %c2 : !quir.cbit<1>
    %c4 = oq3.cbit_not %c3 : !quir.cbit<1>
    return %c4 : !quir.cbit<1>
}

// CHECK: func @t2
func @t2 (%arg1 : !quir.cbit<1>) -> (!quir.cbit<1>) {
// CHECK-NOT: oq3.cbit_not %{{.*}} : !quir.cbit<1>
// CHECK-NOT: oq3.cbit_not %{{.*}} : !quir.cbit<1>
    %c2 = oq3.cbit_not %arg1 : !quir.cbit<1>
    %c3 = oq3.cbit_not %c2 : !quir.cbit<1>
    return %c3 : !quir.cbit<1>
}

// CHECK: func @t7
func @t7 (%arg0 : i32, %arg1 : f32) -> (i32, f32) {
    %r1 = "quir.cast"(%arg0) : (i32) -> i32
    %r2 = "quir.cast"(%arg1) : (f32) -> f32
    // CHECK: return %arg0, %arg1 : i32, f32
    return %r1, %r2 : i32, f32
}

// RUN: qss-compiler --inline -X=mlir %s | FileCheck %s

module {
    func private @proto10(%arg0 : i32) -> i32
    func private @proto11(%arg0 : i32) -> i32
    func private @proto12(%arg0 : i32) -> i32
    func private @proto20(%arg0 : i32, %arg1 : i32) -> i32
    func private @proto21(%arg0 : i32, %arg1 : i32) -> i32
    func private @proto22(%arg0 : i32, %arg1 : i32) -> i32
    func private @proto30(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32
    func private @proto31(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32
    func private @proto32(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32
    func @mul2(%arg0 : i32) -> i32 {
        %1 = arith.constant 2 : i32
        %2 = "arith.muli"(%1, %arg0) : (i32, i32) -> i32
        return %2 : i32
    }
    // CHECK-LABEL: func @mymul
    func @mymul(%arg0 : i32, %arg1 : i32) -> i32 {
        %2 = "oq3.kernel_call"(%arg0) {callee = @mul2} : (i32) -> i32
        %3 = "oq3.kernel_call"(%arg1) {callee = @mul2} : (i32) -> i32
        %4 = "arith.muli"(%2, %3) : (i32, i32) -> i32
        return %4 : i32
    }
    // CHECK-LABEL: func @nest10
    func @nest10(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "oq3.kernel_call"(%arg0) {callee = @proto10} : (i32) -> i32
        %1 = "oq3.kernel_call"(%0, %arg1) {callee = @proto20} : (i32, i32) -> i32
        return %1 : i32
    }
    // CHECK-LABEL: func @nest11
    func @nest11(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "oq3.kernel_call"(%arg0) {callee = @proto11} : (i32) -> i32
        %1 = "oq3.kernel_call"(%0, %arg1) {callee = @proto21} : (i32, i32) -> i32
        return %1 : i32
    }
    // CHECK-LABEL: func @nest12
    func @nest12(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "oq3.kernel_call"(%arg0) {callee = @proto12} : (i32) -> i32
        %1 = "oq3.kernel_call"(%0, %arg1) {callee = @proto22} : (i32, i32) -> i32
        return %1 : i32
    }
    // CHECK-LABEL: func @nest20
    func @nest20(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "oq3.kernel_call"(%arg0, %arg1) {callee = @nest10} : (i32, i32) -> i32
        %1 = "oq3.kernel_call"(%arg0, %0) {callee = @nest11} : (i32, i32) -> i32
        %2 = "oq3.kernel_call"(%1, %0) {callee = @nest12} : (i32, i32) -> i32
        return %2 : i32
    }
    // CHECK-LABEL: func @main
    func @main(%arg0 : i32) -> i32 {
        %a = arith.constant 3 : i32
        %b = arith.constant 5 : i32
        %a2 = "oq3.kernel_call"(%a) {callee = @mul2} : (i32) -> i32
        %a3 = "arith.muli"(%a2, %arg0) : (i32, i32) -> i32
        %a4 = "oq3.kernel_call"(%a3, %b) {callee = @mymul}: (i32, i32) -> i32
        %a5 = "oq3.kernel_call"(%a4, %a3) {callee = @nest10} : (i32, i32) -> i32
        // CHECK:  %{{.*}} = oq3.kernel_call @proto22(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
        %a6 = "oq3.kernel_call"(%a5, %a4) {callee = @nest20} : (i32, i32) -> i32
        return %a6 : i32
    }
}

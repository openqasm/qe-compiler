// RUN: qss-compiler --inline -X=mlir %s | FileCheck %s

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
        %2 = "quir.call_kernel"(%arg0) {callee = @mul2} : (i32) -> i32
        %3 = "quir.call_kernel"(%arg1) {callee = @mul2} : (i32) -> i32
        %4 = "arith.muli"(%2, %3) : (i32, i32) -> i32
        return %4 : i32
    }
    // CHECK-LABEL: func @nest10
    func @nest10(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "quir.call_kernel"(%arg0) {callee = @proto10} : (i32) -> i32
        %1 = "quir.call_kernel"(%0, %arg1) {callee = @proto20} : (i32, i32) -> i32
        return %1 : i32
    }
    // CHECK-LABEL: func @nest11
    func @nest11(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "quir.call_kernel"(%arg0) {callee = @proto11} : (i32) -> i32
        %1 = "quir.call_kernel"(%0, %arg1) {callee = @proto21} : (i32, i32) -> i32
        return %1 : i32
    }
    // CHECK-LABEL: func @nest12
    func @nest12(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "quir.call_kernel"(%arg0) {callee = @proto12} : (i32) -> i32
        %1 = "quir.call_kernel"(%0, %arg1) {callee = @proto22} : (i32, i32) -> i32
        return %1 : i32
    }
    // CHECK-LABEL: func @nest20
    func @nest20(%arg0 : i32, %arg1 : i32) -> i32 {
        %0 = "quir.call_kernel"(%arg0, %arg1) {callee = @nest10} : (i32, i32) -> i32
        %1 = "quir.call_kernel"(%arg0, %0) {callee = @nest11} : (i32, i32) -> i32
        %2 = "quir.call_kernel"(%1, %0) {callee = @nest12} : (i32, i32) -> i32
        return %2 : i32
    }
    // CHECK-LABEL: func @main
    func @main(%arg0 : i32) -> i32 {
        %a = arith.constant 3 : i32
        %b = arith.constant 5 : i32
        %a2 = "quir.call_kernel"(%a) {callee = @mul2} : (i32) -> i32
        %a3 = "arith.muli"(%a2, %arg0) : (i32, i32) -> i32
        %a4 = "quir.call_kernel"(%a3, %b) {callee = @mymul}: (i32, i32) -> i32
        %a5 = "quir.call_kernel"(%a4, %a3) {callee = @nest10} : (i32, i32) -> i32
        // CHECK:  %{{.*}} = quir.call_kernel @proto22(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
        %a6 = "quir.call_kernel"(%a5, %a4) {callee = @nest20} : (i32, i32) -> i32
        return %a6 : i32
    }
}

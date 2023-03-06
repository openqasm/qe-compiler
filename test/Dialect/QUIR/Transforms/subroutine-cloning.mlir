// RUN: qss-compiler -X=mlir --subroutine-cloning %s | FileCheck %s
// RUN: qss-compiler -X=mlir --subroutine-cloning %s --remove-qubit-args | FileCheck %s --check-prefix=NOQARG

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

func @classical() -> i32 {
  %ret = arith.constant 32 : i32
  return %ret : i32
}

func @sub1(%q0 : !quir.qubit<1>) {
  quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
  return
}

func @sub2(%q0 : !quir.qubit<1>, %q1 : !quir.qubit<1>, %a1 : !quir.angle<20>) {
  quir.call_gate @h(%q0) : (!quir.qubit<1>) -> ()
  quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
  quir.call_subroutine @sub1(%q0) : (!quir.qubit<1>) -> ()
  return
}

func @main() -> i32 {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %a1 = quir.constant #quir.angle<3.14159 : !quir.angle<20>>
  quir.call_subroutine @sub1(%q0) : (!quir.qubit<1>) -> ()
  quir.call_subroutine @sub1(%q1) : (!quir.qubit<1>) -> ()
  quir.call_subroutine @sub2(%q0, %q1, %a1) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<20>) -> ()
  quir.call_subroutine @sub2(%q1, %q0, %a1) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<20>) -> ()
  %ret = quir.call_subroutine @classical() : () -> (i32)
  return %ret : i32
}

// CHECK:   func @classical() -> i32 {
// CHECK:   func @sub1_q0(%arg0: !quir.qubit<1> {quir.physicalId = 0 : i32}) {
// CHECK:   func @sub1_q1(%arg0: !quir.qubit<1> {quir.physicalId = 1 : i32}) {
// CHECK:   func @sub2_q0_q1(%arg0: !quir.qubit<1> {quir.physicalId = 0 : i32}, %arg1: !quir.qubit<1> {quir.physicalId = 1 : i32}, %arg2: !quir.angle<20>) {
// CHECK:     quir.call_subroutine @sub1_q0(%arg0) : (!quir.qubit<1>) -> ()
// CHECK:   func @sub2_q1_q0(%arg0: !quir.qubit<1> {quir.physicalId = 1 : i32}, %arg1: !quir.qubit<1> {quir.physicalId = 0 : i32}, %arg2: !quir.angle<20>) {
// CHECK:     quir.call_subroutine @sub1_q1(%arg0) : (!quir.qubit<1>) -> ()
// CHECK:   func @main() -> i32 {
// CHECK:     quir.call_subroutine @sub1_q0(%0) : (!quir.qubit<1>) -> ()
// CHECK:     quir.call_subroutine @sub1_q1(%1) : (!quir.qubit<1>) -> ()
// CHECK:     quir.call_subroutine @sub2_q0_q1(%0, %1, %angle) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<20>) -> ()
// CHECK:     quir.call_subroutine @sub2_q1_q0(%1, %0, %angle) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<20>) -> ()
// CHECK:     %{{[0-9]+}} = quir.call_subroutine @classical() : () -> i32

// NOQARG:   func @classical() -> i32 {
// NOQARG:   func @sub1_q0() {
// NOQARG-NEXT:    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// NOQARG:   func @sub1_q1() {
// NOQARG-NEXT:    %0 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// NOQARG:   func @sub2_q0_q1(%arg0: !quir.angle<20>) {
// NOQARG-NEXT:    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// NOQARG-NEXT:    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// NOQARG:     quir.call_subroutine @sub1_q0() : () -> ()
// NOQARG:   func @sub2_q1_q0(%arg0: !quir.angle<20>) {
// NOQARG-NEXT:    %0 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// NOQARG-NEXT:    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// NOQARG:     quir.call_subroutine @sub1_q1() : () -> ()
// NOQARG:   func @main() -> i32 {
// NOQARG:     quir.call_subroutine @sub1_q0() : () -> ()
// NOQARG:     quir.call_subroutine @sub1_q1() : () -> ()
// NOQARG:     quir.call_subroutine @sub2_q0_q1(%angle) : (!quir.angle<20>) -> ()
// NOQARG:     quir.call_subroutine @sub2_q1_q0(%angle) : (!quir.angle<20>) -> ()
// NOQARG:     %{{.*}} = quir.call_subroutine @classical() : () -> i32

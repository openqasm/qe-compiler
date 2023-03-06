// RUN: qss-compiler -X=mlir %s --quantum-decorate | FileCheck %s

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

func @t1 (%cond : i1) -> () {
  %q0 = quir.declare_qubit {id = 0: i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1: i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2: i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3: i32} : !quir.qubit<1>
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32]}
  }
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32]}
  }
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32, 2 : i32, 3 : i32]}
  }
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    %res = "quir.measure"(%q1) : (!quir.qubit<1>) -> i1
    quir.reset %q0 : !quir.qubit<1>
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32, 2 : i32, 3 : i32]}
  }
  return
}

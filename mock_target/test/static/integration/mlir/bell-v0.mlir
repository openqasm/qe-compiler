// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload | FileCheck %s

// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
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

// CHECK: Manifest
func @main () -> i32 {

  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %a0 = quir.constant #quir.angle<1.57079632679 : !quir.angle<20>>
  %a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
  %a2 = quir.constant #quir.angle<3.14159265359 : !quir.angle<20>>
  quir.builtin_U %q0, %a0, %a1, %a2 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
  quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
  %zero = arith.constant 0 : i32
  return %zero : i32
}

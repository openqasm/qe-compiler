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
    func private @kernel1 (%ca1 : memref<1xi1>, %ca2 : memref<1xi1>, %ca3 : memref<1xi1>) -> memref<1xi1>
    func private @proto (%qa1 : !quir.qubit<1>) -> !quir.qubit<1>
    func @gateCall1(%q1 : !quir.qubit<1>, %lambda : !quir.angle<1>) -> () {
        %zero = quir.constant #quir.angle<0.0 : !quir.angle<1>>
        quir.builtin_U %q1, %zero, %zero, %lambda : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        return
    }
    // CHECK-LABEL: func @bar()
    func @bar() {
        %qa1 = quir.declare_qubit : !quir.qubit<1>
        %qb1 = quir.declare_qubit : !quir.qubit<1>
        %qc1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: quir.builtin_CX %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.qubit<1>
        quir.builtin_CX %qa1, %qb1 : !quir.qubit<1>, !quir.qubit<1>
        // CHECK: %{{.*}} = quir.constant #quir.angle<1.000000e-01 : !quir.angle<1>>
        %theta = quir.constant #quir.angle<0.1 : !quir.angle<1>>
        // CHECK: quir.builtin_U %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        quir.builtin_U %qb1, %theta, %theta, %theta : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        return
    }
}

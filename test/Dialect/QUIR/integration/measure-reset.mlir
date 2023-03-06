// RUN: qss-compiler -X=mlir %s | FileCheck %s

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
    func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qa1 = quir.declare_qubit : !quir.qubit<1>
        %qb1 = quir.declare_qubit : !quir.qubit<1>
        %qc1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: quir.reset %{{.*}} : !quir.qubit<1>
        quir.reset %qa1 : !quir.qubit<1>
        quir.reset %qb1 : !quir.qubit<1>
        quir.reset %qc1 : !quir.qubit<1>
        // CHECK: quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
        %res1 = "quir.measure"(%qb1) : (!quir.qubit<1>) -> i1
        // SYNCH: quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
        // SYNCH-NEXT: qcs.synchronize %{{.*}} : (!quir.qubit<1>) -> ()
        return
    }
}

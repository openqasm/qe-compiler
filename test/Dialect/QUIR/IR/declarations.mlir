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

// This test case checks that QUIR declarations can be parsed from
// textual/assembly input.
module {
    func @bar() {
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qa1 = quir.declare_qubit : !quir.qubit<1>
        %qb1 = quir.declare_qubit : !quir.qubit<1>
        %qc1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qd1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: %{{.*}} = quir.constant #quir.angle<1.000000e-01 : !quir.angle<1>>
        %theta = quir.constant #quir.angle<0.1 : !quir.angle<1>>
        // CHECK: %{{.*}} = quir.constant #quir.angle<2.000000e-01  : !quir.angle>
        %mu = quir.constant #quir.angle<0.2 : !quir.angle>
        // CHECK %{{.*}} = quir.declare_duration {value = "10ns"} : !quir.duration
        %len1 = "quir.declare_duration"() {value = "10ns"} : () -> !quir.duration
        // CHECK %{{.*}} = quir.declare_stretch : !quir.stretch
        %s1 = "quir.declare_stretch"() : () -> !quir.stretch
        // CHECK %{{.*}} = quir.declare_stretch : !quir.stretch
        %s2 = quir.declare_stretch : !quir.stretch
        quir.declare_variable { input } @flags : !quir.cbit<32>
        quir.declare_variable { output } @result : !quir.cbit<1>
        return
    }
}

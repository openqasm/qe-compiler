// RUN: qss-compiler -X=mlir --classical-only-detection %s | FileCheck %s

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

%q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
%ang = quir.constant #quir.angle<0.00 : !quir.angle<20>>
%ang_incr = quir.constant #quir.angle<0.1 : !quir.angle<20>>
%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
// CHECK: %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!quir.angle<20>)
%ang_final = scf.for %iv = %lb to %ub step %step
    iter_args(%ang_iter = %ang) -> (!quir.angle<20>) {
    quir.call_gate @RX(%q1, %ang_iter) : (!quir.qubit<1>, !quir.angle<20>) -> ()
    %ang_sum = quir.angle_add %ang_iter, %ang_incr : !quir.angle<20>
    scf.yield %ang_sum : !quir.angle<20>
}

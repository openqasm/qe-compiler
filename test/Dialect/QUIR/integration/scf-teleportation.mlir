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

// OpenQASM 3.0 Quantum Teleportation example from the spec.
// qubit q[3];
%qa1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
%qb1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
%qc1 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// bit c0;
// bit c1;
// bit c2;
quir.declare_variable @c0 : !quir.cbit<1>
quir.declare_variable @c1 : !quir.cbit<1>
quir.declare_variable @c2 : !quir.cbit<1>
// reset q;
// CHECK: quir.reset %{{.*}} : !quir.qubit<1>
quir.reset %qa1 : !quir.qubit<1>
quir.reset %qb1 : !quir.qubit<1>
quir.reset %qc1 : !quir.qubit<1>
// u3(0.3, 0.2, 0.1) q[0];
// CHECK: %{{.*}} = quir.constant #quir.angle<3.000000e-01 : !quir.angle<1>>
%theta  = quir.constant #quir.angle<0.3 : !quir.angle<1>>
%phi    = quir.constant #quir.angle<0.2 : !quir.angle<1>>
%lambda = quir.constant #quir.angle<0.1 : !quir.angle<1>>
// CHECK: quir.builtin_U %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
quir.builtin_U %qa1, %theta, %phi, %lambda : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
// h q[1];
// CHECK: quir.call_gate @H(%{{.*}}) : (!quir.qubit<1>) -> ()
"quir.call_gate"(%qb1) {callee = @H} : (!quir.qubit<1>) -> ()
// barrier q;
// TODO: unimplemented
// cx q[0], q[1];
quir.builtin_CX %qa1, %qb1 : !quir.qubit<1>, !quir.qubit<1>
// h q[0];
"quir.call_gate"(%qa1) {callee = @H} : (!quir.qubit<1>) -> ()
// c0 = measure q[0];
%zeroind = arith.constant 0 : index
%mres1 = "quir.measure"(%qa1) : (!quir.qubit<1>) -> i1
quir.assign_cbit_bit @c0<1> [0] : i1 = %mres1
// c1 = measure q[1];
%mres2 = "quir.measure"(%qb1) : (!quir.qubit<1>) -> i1
quir.assign_cbit_bit @c1<1> [0] : i1 = %mres2
// if(c0==1) z q[2];
%ca2 = quir.use_variable @c0 : !quir.cbit<1>
%cond1 = "quir.cast"(%ca2) : (!quir.cbit<1>) -> i1
scf.if %cond1 {
    "quir.call_gate"(%qc1) {callee = @Z} : (!quir.qubit<1>) -> ()
} //{quir.classicalOnly = false}
// if(c1==1) { x q[2]; } // braces optional in this case
%cb2 = quir.use_variable @c1 : !quir.cbit<1>
%cond2 = "quir.cast"(%cb2) : (!quir.cbit<1>) -> i1
scf.if %cond2 {
    "quir.call_gate"(%qc1) {callee = @X} : (!quir.qubit<1>) -> ()
} //{quir.classicalOnly = false}
// post q[2]; // NOP/identity
// c2 = measure q[2];
%mres3 = "quir.measure"(%qc1) : (!quir.qubit<1>) -> i1
quir.assign_cbit_bit @c2<1> [0] : i1 = %mres3

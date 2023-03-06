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

// CHECK: [[a0:%angle[_0-9]*]] = quir.constant #quir.angle<0.000000e+00 : !quir.angle<20>>
%a0 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
// CHECK: [[a1:%angle[_0-9]*]] = quir.constant #quir.angle<0.000000e+00 : !quir.angle<20>>
%a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
// CHECK-COUNT-10: %{{.*}} = quir.angle_cmp {predicate = "{{.*}}"} [[a0]], [[a1]] : !quir.angle<20> -> i1
%b0 = quir.angle_cmp {predicate = "eq"} %a0, %a1 : !quir.angle<20> -> i1
%b1 = quir.angle_cmp {predicate = "ne"} %a0, %a1 : !quir.angle<20> -> i1
%b2 = quir.angle_cmp {predicate = "slt"} %a0, %a1 : !quir.angle<20> -> i1
%b3 = quir.angle_cmp {predicate = "sle"} %a0, %a1 : !quir.angle<20> -> i1
%b4 = quir.angle_cmp {predicate = "sgt"} %a0, %a1 : !quir.angle<20> -> i1
%b5 = quir.angle_cmp {predicate = "sge"} %a0, %a1 : !quir.angle<20> -> i1
%b6 = quir.angle_cmp {predicate = "ult"} %a0, %a1 : !quir.angle<20> -> i1
%b7 = quir.angle_cmp {predicate = "ule"} %a0, %a1 : !quir.angle<20> -> i1
%b8 = quir.angle_cmp {predicate = "ugt"} %a0, %a1 : !quir.angle<20> -> i1
%b9 = quir.angle_cmp {predicate = "uge"} %a0, %a1 : !quir.angle<20> -> i1

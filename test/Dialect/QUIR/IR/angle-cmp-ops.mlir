// RUN: qss-compiler -X=mlir %s | FileCheck %s

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// CHECK: [[a0:%angle[_0-9]*]] = quir.constant #quir.angle<0.000000e+00 : !quir.angle<20>>
%a0 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
// CHECK: [[a1:%angle[_0-9]*]] = quir.constant #quir.angle<0.000000e+00 : !quir.angle<20>>
%a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
// CHECK-COUNT-10: %{{.*}} = oq3.angle_cmp {predicate = "{{.*}}"} [[a0]], [[a1]] : !quir.angle<20> -> i1
%b0 = oq3.angle_cmp {predicate = "eq"} %a0, %a1 : !quir.angle<20> -> i1
%b1 = oq3.angle_cmp {predicate = "ne"} %a0, %a1 : !quir.angle<20> -> i1
%b2 = oq3.angle_cmp {predicate = "slt"} %a0, %a1 : !quir.angle<20> -> i1
%b3 = oq3.angle_cmp {predicate = "sle"} %a0, %a1 : !quir.angle<20> -> i1
%b4 = oq3.angle_cmp {predicate = "sgt"} %a0, %a1 : !quir.angle<20> -> i1
%b5 = oq3.angle_cmp {predicate = "sge"} %a0, %a1 : !quir.angle<20> -> i1
%b6 = oq3.angle_cmp {predicate = "ult"} %a0, %a1 : !quir.angle<20> -> i1
%b7 = oq3.angle_cmp {predicate = "ule"} %a0, %a1 : !quir.angle<20> -> i1
%b8 = oq3.angle_cmp {predicate = "ugt"} %a0, %a1 : !quir.angle<20> -> i1
%b9 = oq3.angle_cmp {predicate = "uge"} %a0, %a1 : !quir.angle<20> -> i1

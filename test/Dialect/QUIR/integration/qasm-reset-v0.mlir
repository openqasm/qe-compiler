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


// CHECK: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
%q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: quir.reset %{{.*}} : !quir.qubit<1>
quir.reset %q0 : !quir.qubit<1>

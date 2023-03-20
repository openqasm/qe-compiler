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


// gate h q {
//     U(1.57079632679, 0.0, 3.14159265359) q;
// }
func @h (%q : !quir.qubit<1>) -> () {
    %a0 = quir.constant #quir.angle<1.57079632679 : !quir.angle<20>>
    %a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
    %a2 = quir.constant #quir.angle<3.14159265359 : !quir.angle<20>>
    quir.builtin_U %q, %a0, %a1, %a2 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
    return
}

func @cx (%ctrl : !quir.qubit<1>, %targ : !quir.qubit<1>) {
    quir.builtin_CX %ctrl, %targ : !quir.qubit<1>, !quir.qubit<1>
    return
}

func @fake_multiqubit (%ctrl : !quir.qubit<1>, %targ : !quir.qubit<1>, %p1 : !quir.angle<20>) {
    quir.builtin_U %ctrl, %p1, %p1, %p1 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
    quir.builtin_CX %ctrl, %targ : !quir.qubit<1>, !quir.qubit<1>
    return
}

func @main () -> i32 {
    // qubit %0;
    // qubit %1;
    // CHECK: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    // h %0;
    quir.call_gate @h(%q0) : (!quir.qubit<1>) -> ()
    // CX %0, %1;
    quir.call_gate @cx(%q0, %q1) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    // measure %0 -> c0 and measure %1 -> c1 only in qasm-bell-v0.mlir
    %zero = arith.constant 0 : i32
    return %zero : i32
}

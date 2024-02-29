// RUN: qss-compiler --merge-circuits %s | FileCheck %s
//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// This test validates the fix for a bug when merging circuits with non-unique arguments

module {
  quir.circuit @circuit_0_q0_q1(%arg0: !quir.qubit<1> {quir.physicalId = 0 : i32}, %arg1: !quir.qubit<1> {quir.physicalId = 1 : i32}, %arg2: !quir.duration<dt>, %arg3: !quir.duration<dt>) attributes { quir.physicalIds = [0 : i32, 1 : i32]} {
    quir.delay %arg2, (%arg0) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    quir.delay %arg3, (%arg1) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
    quir.return
  }
  quir.circuit @circuit_1_q2(%arg0: !quir.qubit<1> {quir.physicalId = 7 : i32}) -> i1 attributes {quir.physicalIds = [2 : i32]} {
    %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
    quir.return %0 : i1
  }
  func.func @main() -> i32  {
    %c0_i32 = arith.constant 0 : i32
    %dur = quir.constant #quir.duration<3.750080e+05> : !quir.duration<dt>

    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    quir.call_circuit @circuit_0_q0_q1(%0, %1, %dur, %dur) : (!quir.qubit<1>, !quir.qubit<1>, !quir.duration<dt>, !quir.duration<dt>) -> ()
    quir.barrier %2, %0, %1 : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()
    %28 = quir.call_circuit @circuit_1_q2(%2) : (!quir.qubit<1>) -> i1
    // CHECK: %3 = quir.call_circuit @circuit_0_q0_q1_circuit_1_q2(%0, %1, %dur, %dur, %2)

    return %c0_i32 : i32
  }
}

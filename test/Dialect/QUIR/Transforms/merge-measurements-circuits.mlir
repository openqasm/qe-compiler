// RUN: qss-compiler -X=mlir --enable-circuits=true --subroutine-cloning --merge-measures-topological %s | FileCheck %s

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


module {
quir.circuit @circuit_0(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) -> (i1, i1) {
  %0 = quir.measure(%arg0) : (!quir.qubit<1>) -> i1
  %1 = quir.measure(%arg1) : (!quir.qubit<1>) -> i1
  quir.return %0, %1: i1, i1
  // CHECK: [[MEASURE:%.*]]:2 = quir.measure(%arg0, %arg1) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  // CHECK: quir.return [[MEASURE]]#0, [[MEASURE]]#1 : i1, i1
}
func @main() -> i32 {
  %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %2:2 = quir.call_circuit @circuit_0(%0, %1) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
}

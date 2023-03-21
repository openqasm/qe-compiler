// RUN: qss-compiler -X=mlir --canonicalize --reorder-measures %s | FileCheck %s

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

// CHECK: func @three
func @three(%c : memref<1xi1>, %ind : index, %angle_0 : !quir.angle<64>) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  quir.call_gate @rz(%q0, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  quir.call_gate @sx(%q0) : (!quir.qubit<1>) -> ()
  quir.call_gate @rz(%q0, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  quir.call_gate @rz(%q1, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  quir.call_gate @sx(%q1) : (!quir.qubit<1>) -> ()
  quir.call_gate @rz(%q1, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  quir.call_gate @rz(%q2, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  quir.call_gate @sx(%q2) : (!quir.qubit<1>) -> ()
  quir.call_gate @rz(%q2, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
// CHECK: [[Q00:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: [[Q01:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// CHECK: [[Q02:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// CHECK:          quir.call_gate @rz([[Q00]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     quir.call_gate @sx([[Q00]]) : (!quir.qubit<1>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q00]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q01]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     quir.call_gate @sx([[Q01]]) : (!quir.qubit<1>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q01]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q02]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     quir.call_gate @sx([[Q02]]) : (!quir.qubit<1>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q02]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     [[RES00:%.*]] = quir.measure([[Q00]]) : (!quir.qubit<1>) -> i1
// CHECK-NEXT:     memref.store [[RES00]], {{%.*}}[{{%.*}}] : memref<1xi1>
// CHECK-NEXT:     [[RES01:%.*]] = quir.measure([[Q01]]) : (!quir.qubit<1>) -> i1
// CHECK-NEXT:     memref.store [[RES01]], {{%.*}}[{{%.*}}] : memref<1xi1>
// CHECK-NEXT:     [[RES02:%.*]] = quir.measure([[Q02]]) : (!quir.qubit<1>) -> i1
  return
}

// Reordering should fail when there is a usage of the same qubit in a gate
// after the measurement
// CHECK: func @reorder_fail1
func @reorder_fail1(%c : memref<1xi1>, %ind : index, %angle_0 : !quir.angle<64>) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  quir.call_gate @rz(%q0, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  quir.call_gate @sx(%q0) : (!quir.qubit<1>) -> ()
  quir.call_gate @rz(%q0, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  quir.call_gate @rz(%q1, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  quir.call_gate @sx(%q0) : (!quir.qubit<1>) -> ()
  quir.call_gate @rz(%q1, %angle_0) : (!quir.qubit<1>, !quir.angle<64>) -> ()
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
// CHECK: [[Q10:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: [[Q11:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// CHECK: [[Q12:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// CHECK:          quir.call_gate @rz([[Q10]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     quir.call_gate @sx([[Q10]]) : (!quir.qubit<1>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q10]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q11]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     [[RES10:%.*]] = quir.measure([[Q10]]) : (!quir.qubit<1>) -> i1
// CHECK-NEXT:     memref.store [[RES10]], {{%.*}}[{{%.*}}] : memref<1xi1>
// CHECK-NEXT:     quir.call_gate @sx([[Q10]]) : (!quir.qubit<1>) -> ()
// CHECK-NEXT:     quir.call_gate @rz([[Q11]], {{%.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// CHECK-NEXT:     [[RES11:%.*]] = quir.measure([[Q11]]) : (!quir.qubit<1>) -> i1
// CHECK-NEXT:     memref.store [[RES11]], {{%.*}}[{{%.*}}] : memref<1xi1>
  return
}

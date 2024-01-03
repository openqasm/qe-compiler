// RUN: qss-compiler -X=mlir --canonicalize --merge-measures-lexographical %s | FileCheck %s --check-prefix LEX
// RUN: qss-compiler -X=mlir --canonicalize --merge-measures-topological %s | FileCheck %s --check-prefix TOP

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

func.func @one() {
  %q = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  // LEX:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  // TOP:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res = quir.measure(%q) : (!quir.qubit<1>) -> (i1)
  return
}

func.func @two(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  // LEX:  %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  // TOP:  %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  return
}

func.func @three(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  // LEX:  %{{.*}}:3 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1)
  // TOP:  %{{.*}}:3 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  return
}

func.func @four(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
  // LEX:  %{{.*}}:4 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
  // TOP:  %{{.*}}:4 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  memref.store %res2, %c[%ind] : memref<1xi1>
  %res3 = quir.measure(%q3) : (!quir.qubit<1>) -> (i1)
  return
}

func.func @four_interrupted(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
  %q4 = quir.declare_qubit {id = 4 : i32} : !quir.qubit<1>

  // LEX:  %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  // TOP:  %{{.*}}:4 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>

  // LEX: quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
  // TOP: quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
  quir.call_gate @x(%q4) : (!quir.qubit<1>) -> ()

  // LEX:  %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  // TOP-NOT:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> (i1)
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  memref.store %res2, %c[%ind] : memref<1xi1>
  %res3 = quir.measure(%q3) : (!quir.qubit<1>) -> (i1)
  return
}

func.func @inter_if(%c : memref<1xi1>, %ind : index, %cond : i1) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>

  // LEX: %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  // TOP: %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  scf.if %cond {
    quir.call_gate @cx(%q2, %q3) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
  }

  // LEX: %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  // TOP: %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  return
}

func.func @barrier(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>

  // LEX:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  // TOP:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  quir.barrier %q1, %q0 : (!quir.qubit<1>, !quir.qubit<1>) -> ()

  // LEX:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  // TOP: %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)

  quir.barrier %q0, %q3: (!quir.qubit<1>, !quir.qubit<1>) -> ()

  // LEX:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  // TOP-NOT:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  return
}


func.func @inter_switch(%flag : i32) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
  %q4 = quir.declare_qubit {id = 4 : i32} : !quir.qubit<1>
  %q5 = quir.declare_qubit {id = 5 : i32} : !quir.qubit<1>

  // LEX: %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  // TOP:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)

  quir.switch %flag {
        quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    } [
        0: {
            quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
        }
        1: {
            quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
        }
  ]

  // LEX: %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  // TOP:  %{{.*}}:4 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
  %res1 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  %res2 = quir.measure(%q3) : (!quir.qubit<1>) -> (i1)


  quir.switch %flag {} [
        0: {
            quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
        }
  ]

  // LEX: %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  %res3 = quir.measure(%q4) : (!quir.qubit<1>) -> (i1)
  %res4 = quir.measure(%q5) : (!quir.qubit<1>) -> (i1)
  return
}

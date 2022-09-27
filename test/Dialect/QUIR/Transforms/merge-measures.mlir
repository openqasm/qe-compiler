// RUN: qss-compiler -X=mlir --canonicalize --merge-measures %s | FileCheck %s

// CHECK: func @one
func @one() {
  %q = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  // CHECK:  %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res = quir.measure(%q) : (!quir.qubit<1>) -> (i1)
  return
}

// CHECK: func @two
func @two(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  // CHECK:  %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  return
}

// CHECK: func @three
func @three(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  // CHECK:  %{{.*}}:3 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  return
}

// CHECK: func @four
func @four(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
  // CHECK:  %{{.*}}:4 = quir.measure(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> (i1, i1, i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  memref.store %res2, %c[%ind] : memref<1xi1>
  %res3 = quir.measure(%q3) : (!quir.qubit<1>) -> (i1)
  return
}

// CHECK: func @four_interrupted
func @four_interrupted(%c : memref<1xi1>, %ind : index) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
  // CHECK:  %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  // CHECK: quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
  // CHECK:  %{{.*}}:2 = quir.measure(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
  %res2 = quir.measure(%q2) : (!quir.qubit<1>) -> (i1)
  memref.store %res2, %c[%ind] : memref<1xi1>
  %res3 = quir.measure(%q3) : (!quir.qubit<1>) -> (i1)
  return
}

func @inter_if(%c : memref<1xi1>, %ind : index, %cond : i1) {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  // CHECK: %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res0 = quir.measure(%q0) : (!quir.qubit<1>) -> (i1)
  memref.store %res0, %c[%ind] : memref<1xi1>
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
  }
  // CHECK: %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
  %res1 = quir.measure(%q1) : (!quir.qubit<1>) -> (i1)
  memref.store %res1, %c[%ind] : memref<1xi1>
  return
}

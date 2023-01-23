// RUN: qss-compiler -X=mlir --canonicalize --quir-eliminate-variables %s --canonicalize | FileCheck %s
//
// This test verifies that there is no store-forwarding where control-flow makes
// it impossible.

// CHECK: module
module {
  oq3.declare_variable @b : !quir.cbit<1>
  func @x(%arg0: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    // CHECK-DAG: [[MEMREF:%.*]] = memref.alloca() : memref<i1>
    // CHECK-DAG: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32}
    // CHECK-DAG: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32}
    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

    %false = arith.constant false
    %4 = "quir.cast"(%false) : (i1) -> !quir.cbit<1>
    oq3.assign_variable @b : !quir.cbit<1> = %4

    // CHECK: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]])
    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1

    // CHECK: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]])
    // CHECK: affine.store [[MEASURE1]], [[MEMREF]]
    %6 = quir.measure(%2) : (!quir.qubit<1>) -> i1
    quir.assign_cbit_bit @b<1> [0] : i1 = %6

    // A variable update inside a control flow branch currently cannot be
    // simplified. Thus the store and load operations must be kept.
    scf.if %5 {
      quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
      quir.assign_cbit_bit @b<1> [0] : i1 = %5
      %cst = constant unit
    }

    // CHECK: [[LOAD:%.*]] = affine.load [[MEMREF]]
    %10 = oq3.use_variable @b : !quir.cbit<1>
    %c1_i32_1 = arith.constant 1 : i32
    %11 = "quir.cast"(%10) : (!quir.cbit<1>) -> i32

    %12 = arith.cmpi eq, %11, %c1_i32_1 : i32
    // CHECK: scf.if [[LOAD]]
    scf.if %12 {
      quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
      %cst = constant unit
    }

    return %11 : i32
  }
}

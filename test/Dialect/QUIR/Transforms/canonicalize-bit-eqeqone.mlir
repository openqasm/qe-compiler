// RUN: qss-compiler -X=mlir --canonicalize %s | FileCheck %s --implicit-check-not extui --implicit-check-not cmpi --implicit-check-not cast
//
// This test case validates that comparisons between zero-extended i1 and
// constant 1 for equality are simplified.

// CHECK: module
module {
  func @extract(%in : !quir.cbit<2>) -> i1 {
    // CHECK: [[BIT:%.]] = oq3.cbit_extractbit
    %1 = oq3.cbit_extractbit(%in : !quir.cbit<2>) [0] : i1
    %c1_i32 = arith.constant 1 : i32
    %2 = "oq3.cast"(%1): (i1) -> i32
    %3 = arith.cmpi eq, %2, %c1_i32 : i32
    // CHECK: return [[BIT]] : i1
    return %3 : i1
  }

  func @main() -> i1 {
    %c1_i32 = arith.constant 1 : i32

    %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    // CHECK: [[MEASURE:%.*]] = quir.measure
    %2 = quir.measure(%1) : (!quir.qubit<1>) -> i1

    %3 = arith.extui %2 : i1 to i32
    %4 = arith.cmpi eq, %3, %c1_i32 : i32

    // CHECK: return [[MEASURE]]
    return %4 : i1
  }
}

// RUN: qss-compiler --canonicalize --reorder-measures %s | FileCheck %s

// CHECK: module
module {
  oq3.declare_variable @results : !quir.cbit<1>
  func @cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    %4 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    %5 = "oq3.cast"(%4) : (i1) -> !quir.cbit<1>
    oq3.assign_variable @results : !quir.cbit<1> = %5
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    quir.builtin_CX %1, %2 : !quir.qubit<1>, !quir.qubit<1>
    %6 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    %7 = "oq3.cast"(%6) : (i1) -> !quir.cbit<1>
    oq3.assign_variable @results : !quir.cbit<1> = %7
    return %c0_i32 : i32
  }
}

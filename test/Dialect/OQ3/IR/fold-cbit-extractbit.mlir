// RUN: qss-opt %s --canonicalize | qss-opt | FileCheck %s --implicit-check-not cbit_extractbit
// Verify that all oq3.cbit_extractbit operations are eliminated

// CHECK: func @single_bit(%[[ARG0:.*]]: i1) -> i1 {
func @single_bit(%bit: i1) -> i1 {
    %2 = oq3.cbit_extractbit(%bit : i1) [0] : i1
    // CHECK: return %[[ARG0]] : i1
    return %2 : i1
}

// CHECK: func @two_bits(%[[ARG0:.*]]: !quir.cbit<2>, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: i1)
func @two_bits(%cbit: !quir.cbit<2>, %bit1: i1, %bit2: i1) -> i1 {
   %0 = oq3.cbit_insertbit(%cbit : !quir.cbit<2>)[0] = %bit1 : !quir.cbit<2>
   %1 = oq3.cbit_insertbit(%cbit : !quir.cbit<2>)[1] = %bit2 : !quir.cbit<2>

   %2 = oq3.cbit_extractbit(%1 : !quir.cbit<2>) [1] : i1
   // CHECK: return %[[ARG2]] : i1
   return %2 : i1
}

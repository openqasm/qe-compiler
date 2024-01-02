OPENQASM 3.0;
// RUN: qss-compiler %s --emit=mlir --oq3-limit-cbit-width --canonicalize | FileCheck %s
//
// Test that the we can successfully split cbit arrays larger than 32 bits wide

cbit[4] meas;
// CHECK: oq3.declare_variable @meas : !quir.cbit<4>
cbit[48] wide;
// CHECK-NOT: oq3.declare_variable @wide : !quir.cbit<48>
// CHECK: oq3.declare_variable @wide_00 : !quir.cbit<32>
// CHECK: oq3.declare_variable @wide_1 : !quir.cbit<16>
// test for existing Symbol with new name
cbit[2] wide_0;
// CHECK: oq3.declare_variable @wide_0 : !quir.cbit<2>

// CHECK: func.func @main() -> i32

// CHECK-DAG: %c-8_i4 = arith.constant -8 : i4

// CHECK-NOT: oq3.variable_assign @wide : !quir.cbit<48> = %1
// CHECK-NOT: %2 = "oq3.cast"(%c0_i2) : (i2) -> !quir.cbit<2>

// CHECK: [[CAST0:%.*]] = "oq3.cast"(%c0_i32) : (i32) -> !quir.cbit<32>
// CHECK: oq3.variable_assign @wide_00 : !quir.cbit<32> = [[CAST0]]
// CHECK: [[CAST1:%.*]] = "oq3.cast"(%c0_i16) : (i16) -> !quir.cbit<16>
// CHECK: oq3.variable_assign @wide_1 : !quir.cbit<16> = [[CAST1]]


qubit $0;
// CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $3;
// CHECK: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
qubit $47;
// CHECK: [[QUBIT432:%.*]] = quir.declare_qubit {id = 47 : i32} : !quir.qubit<1>


meas[3] = measure $3;
// CHECK: [[M3:%.*]] = quir.measure([[QUBIT3]]) : (!quir.qubit<1>) -> i1
// CHECK: oq3.cbit_assign_bit @meas<4> [3] : i1 = [[M3]]

wide[0] = measure $0;
// CHECK: [[M0:%.*]] = quir.measure([[QUBIT0]]) : (!quir.qubit<1>) -> i1
// CHECK: oq3.cbit_assign_bit @wide_00<32> [0] : i1 = [[M0]]

wide[47] = measure $47;
// CHECK: [[M47:%.*]] = quir.measure([[QUBIT432]]) : (!quir.qubit<1>) -> i1
// CHECK: oq3.cbit_assign_bit @wide_1<16> [15] : i1 = [[M47]]

// there is a bug with the following line - the initializer value is being
// lost and set to 0
// cbit[36] assignment0  = 34359738368; // 1 << 35

cbit[36] assignment1 = "100000000000000000000000000000000000";

// CHECK: [[LOWER:%.*]] = "oq3.cast"(%c0_i32) : (i32) -> !quir.cbit<32>
// CHECK: oq3.variable_assign @assignment1_0 : !quir.cbit<32> = [[LOWER]]
// CHECK: [[UPPER:%.*]] = "oq3.cast"(%c-8_i4) : (i4) -> !quir.cbit<4>
// CHECK: oq3.variable_assign @assignment1_1 : !quir.cbit<4> = [[UPPER]]

if (assignment1[35] == 1) {
  assignment1[35] = assignment1[0];
}

// CHECK: [[LOAD:%.*]] = oq3.variable_load @assignment1_1 : !quir.cbit<4>
// CHECK: [[COND:%.*]] = oq3.cbit_extractbit([[LOAD]] : !quir.cbit<4>) [3] : i1
// CHECK: scf.if [[COND]] {
// CHECK: [[LOADVAR:%.*]] = oq3.variable_load @assignment1_0 : !quir.cbit<32>
// CHECK: [[LOADBIT:%.*]] = oq3.cbit_extractbit([[LOADVAR]] : !quir.cbit<32>) [0] : i1
// CHECK: oq3.cbit_assign_bit @assignment1_1<4> [3] : i1 = [[LOADBIT]]




OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// MLIR: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;
bit bit_one = 1;
bit bit_two = 0;

bit[2] bitstring = "01";

// Test for ==
// MLIR-DAG: [[LHS:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[RHS:%.*]] = oq3.use_variable @bit_two : !quir.cbit<1>
// MLIR-DAG: [[LHSCAST:%.*]] = "oq3.cast"([[LHS]]) : (!quir.cbit<1>) -> i1
// MLIR-DAG: [[RHSCAST:%.*]] = "oq3.cast"([[RHS]]) : (!quir.cbit<1>) -> i1
// MLIR: [[CMP:%.*]] = arith.cmpi eq, [[LHSCAST]], [[RHSCAST]] : i1
// MLIR: scf.if [[CMP]] {
if (bit_one == bit_two){
    U(0, 0, 0) $0;
}

// MLIR-DAG: [[BITSTRING:%.*]] = oq3.use_variable @bitstring : !quir.cbit<2>
// MLIR-DAG: [[LHS:%.*]] = oq3.cbit_extractbit([[BITSTRING]] : !quir.cbit<2>) [1] : i1
// MLIR-DAG: [[BIT_ONE:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[RHS:%.*]] = "oq3.cast"([[BIT_ONE]]) : (!quir.cbit<1>) -> i1
// MLIR: [[CMP:%.*]] = arith.cmpi eq, [[LHS]], [[RHS]] : i1
// MLIR: scf.if [[CMP]] {
if (bitstring[1] == bit_one) {
    U(0, 0, 0) $0;
}

// TODO what is the semantic here?
//
// Test for <
// MLIR-DAG: [[BIT_ONE:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[BIT_TWO:%.*]] = oq3.use_variable @bit_two : !quir.cbit<1>
// MLIR-DAG: [[LHS:%.*]] = "oq3.cast"([[BIT_ONE]]) : (!quir.cbit<1>) -> i1
// MLIR-DAG: [[RHS:%.*]] = "oq3.cast"([[BIT_TWO]]) : (!quir.cbit<1>) -> i1
// MLIR: [[CMP:%.*]] = arith.cmpi slt, [[LHS]], [[RHS]] : i1
// MLIR: scf.if [[CMP]] {
if (bit_one < bit_two){
    U(0, 0, 0) $0;
}

// Test for >
// MLIR-DAG: [[BIT_ONE:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[BIT_TWO:%.*]] = oq3.use_variable @bit_two : !quir.cbit<1>
// MLIR-DAG: [[LHS:%.*]] = "oq3.cast"([[BIT_TWO]]) : (!quir.cbit<1>) -> i1
// MLIR-DAG: [[RHS:%.*]] = "oq3.cast"([[BIT_ONE]]) : (!quir.cbit<1>) -> i1
// MLIR: [[CMP:%.*]] = arith.cmpi sgt, [[LHS]], [[RHS]] : i1
// MLIR: scf.if [[CMP]] {
if (bit_two > bit_one){
    U(0, 0, 0) $0;
}

// Test for >=
// MLIR-DAG: [[BIT_ONE:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[BIT_TWO:%.*]] = oq3.use_variable @bit_two : !quir.cbit<1>
// MLIR-DAG: [[LHS:%.*]] = "oq3.cast"([[BIT_TWO]]) : (!quir.cbit<1>) -> i1
// MLIR-DAG: [[RHS:%.*]] = "oq3.cast"([[BIT_ONE]]) : (!quir.cbit<1>) -> i1
// MLIR: [[CMP:%.*]] = arith.cmpi sge, [[LHS]], [[RHS]] : i1
// MLIR: scf.if [[CMP]] {
if (bit_two >= bit_one){
    U(0, 0, 0) $0;
}

// Test for <=
// MLIR-DAG: [[BIT_ONE:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[BIT_TWO:%.*]] = oq3.use_variable @bit_two : !quir.cbit<1>
// MLIR-DAG: [[LHS:%.*]] = "oq3.cast"([[BIT_ONE]]) : (!quir.cbit<1>) -> i1
// MLIR-DAG: [[RHS:%.*]] = "oq3.cast"([[BIT_TWO]]) : (!quir.cbit<1>) -> i1
// MLIR: [[CMP:%.*]] = arith.cmpi sle, [[LHS]], [[RHS]] : i1
// MLIR: scf.if [[CMP]] {
if (bit_one <= bit_two){
    U(0, 0, 0) $0;
}

// Test for !=
// MLIR-DAG: [[BIT_ONE:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[BIT_TWO:%.*]] = oq3.use_variable @bit_two : !quir.cbit<1>
// MLIR-DAG: [[LHS:%.*]] = "oq3.cast"([[BIT_TWO]]) : (!quir.cbit<1>) -> i1
// MLIR-DAG: [[RHS:%.*]] = "oq3.cast"([[BIT_ONE]]) : (!quir.cbit<1>) -> i1
// MLIR: [[CMP:%.*]] = arith.cmpi ne, [[LHS]], [[RHS]] : i1
// MLIR: scf.if [[CMP]] {
if (bit_two != bit_one){
    U(0, 0, 0) $0;
}

// MLIR-DAG: [[BIT_ONE:%.*]] = oq3.use_variable @bit_one : !quir.cbit<1>
// MLIR-DAG: [[LHS:%.*]] = "oq3.cast"([[BIT_ONE]]) : (!quir.cbit<1>) -> i1
// MLIR-DAG: [[BITSTRING:%.*]] = oq3.use_variable @bitstring : !quir.cbit<2>
// MLIR-DAG: [[RHS:%.*]] = oq3.cbit_extractbit([[BITSTRING]] : !quir.cbit<2>) [0] : i1
// MLIR: [[CMP:%.*]] = arith.cmpi ne, [[LHS]], [[RHS]] : i1
// MLIR: scf.if [[CMP]] {
if (bit_one != bitstring[0]) {
    U(0, 0, 0) $0;
}

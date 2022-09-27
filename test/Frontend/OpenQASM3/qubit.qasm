OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

qubit $0;
qubit $1;

bit[10] result;
// MLIR-DAG: quir.declare_variable @result : !quir.cbit<10>
// MLIR-DAG: quir.declare_variable @cbits : !quir.cbit<10>
result[4] = measure $0;
result[5] = measure $1;

// MLIR: %[[LOAD1:.*]] = quir.use_variable @result : !quir.cbit<10>
// MLIR: %[[BIT4:.*]] = quir.cbit_extractbit(%[[LOAD1]] : !quir.cbit<10>) [4] : i1
// MLIR: %[[LOAD2:.*]] = quir.use_variable @result : !quir.cbit<10>
// MLIR: %[[BIT5:.*]] = quir.cbit_extractbit(%[[LOAD2]] : !quir.cbit<10>) [5] : i1
// MLIR: %[[CMP1:.*]] = arith.cmpi eq, %[[BIT4]], %[[BIT5]] : i1
// MLIR: scf.if %[[CMP1]] {
if (result[4]==result[5]) {
    U(0, 0, 0) $0;
}

result[4] = measure $0;

// MLIR: %c2_i32 = arith.constant 2 : i32
// MLIR: %c100_i32 = arith.constant 100 : i32
// MLIR: %[[CMP2:.*]] = arith.cmpi slt, %c2_i32, %c100_i32 : i32
// MLIR: scf.if %[[CMP2]] {
if (2 < 100) {
    U(0, 0, 0) $0;
}

// Always re-load classical bits
// MLIR: %[[LOAD3:.*]] = quir.use_variable @result : !quir.cbit<10>
// MLIR-DAG: %[[BIT4:.*]] = quir.cbit_extractbit(%[[LOAD3]] : !quir.cbit<10>) [4] : i1
// MLIR-DAG: %[[CONST1:.*]] = arith.constant 1 : i32
// MLIR-DAG: %[[CAST3:.*]] = "quir.cast"(%[[BIT4]]) : (i1) -> i32
// MLIR: %[[CMP3:.*]] = arith.cmpi ne, %[[CAST3]], %[[CONST1]] : i32
// MLIR: scf.if %[[CMP3]] {
if (result[4] != 1) {
    U(0, 0, 0) $0;
}

// MLIR: %[[LOAD4:.*]] = quir.use_variable @result : !quir.cbit<10>
// MLIR: %[[BIT5:.*]] = quir.cbit_extractbit(%[[LOAD4]] : !quir.cbit<10>) [5] : i1
// MLIR: %[[CAST4:.*]] = "quir.cast"(%[[BIT5]]) : (i1) -> i32
// MLIR: %[[CMP4:.*]] = arith.cmpi eq, %c0_i32_{{.*}}, %[[CAST4]] : i32
// MLIR: scf.if %[[CMP4]] {
if (0 == result[5]) {
    U(0, 0, 0) $0;
}

// MLIR: %[[LOAD5:.*]] = quir.use_variable @result : !quir.cbit<10>
// MLIR: %{{.*}} = "quir.cast"(%[[LOAD5]]) : (!quir.cbit<10>) -> i32
if (result == 1) {
    U(0, 0, 0) $0;
}

// MLIR-DAG: %[[CBITS:.*]] = quir.use_variable @cbits : !quir.cbit<10>
// MLIR-DAG: %[[RESULT:.*]] = quir.use_variable @result : !quir.cbit<10>
// MLIR-DAG: %[[CAST5:.*]] = "quir.cast"(%[[RESULT]]) : (!quir.cbit<10>) -> i10
// MLIR-DAG: %[[CAST6:.*]] = "quir.cast"(%[[CBITS]]) : (!quir.cbit<10>) -> i10
// MLIR: %[[CMP5:.*]] = arith.cmpi eq, %[[CAST5]], %[[CAST6]] : i10
bit[10] cbits;
if (result == cbits) {
  U(0, 0, 0) $0;
}

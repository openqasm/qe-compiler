OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

qubit $0;
bool a = true;
bool b = false;

// Test for ==
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi eq, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a == b){
    U(0, 0, 0) $0;
}

// Test for <
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi slt, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a < b){
    U(0, 0, 0) $0;
}

// Test for >
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi sgt, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a > b){
    U(0, 0, 0) $0;
}

// Test for >=
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi sge, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a >= b){
    U(0, 0, 0) $0;
}

// Test for <=
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi sle, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a <= b){
    U(0, 0, 0) $0;
}

// Test for !=
// MLIR: [[A:%.*]] = quir.use_variable @a : i1
// MLIR: [[B:%.*]] = quir.use_variable @b : i1
// MLIR: [[CMP:%.*]] = arith.cmpi ne, [[A]], [[B]] : i1
// MLIR: scf.if [[CMP:%.*]] {
if (a != b){
    U(0, 0, 0) $0;
}

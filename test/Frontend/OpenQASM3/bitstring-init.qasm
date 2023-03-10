OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

// MLIR-DAG: oq3.declare_variable @noninitialized : !quir.cbit<4>
// MLIR-DAG: oq3.declare_variable @bitstring : !quir.cbit<4>
// MLIR-DAG: oq3.declare_variable @b : !quir.cbit<8>
// MLIR-DAG: oq3.declare_variable @c : !quir.cbit<4>


// MLIR: [[CONST0:%.*]] = arith.constant 0 : i4
// MLIR: [[CAST0:%.*]] = "oq3.cast"([[CONST0]]) : (i4) -> !quir.cbit<4>
// MLIR: oq3.variable_assign @noninitialized : !quir.cbit<4> = [[CAST0]]
cbit[4] noninitialized;

// MLIR: [[CONST6:%.*]] = arith.constant 6 : i4
// MLIR: [[CAST6:%.*]] = "oq3.cast"([[CONST6]]) : (i4) -> !quir.cbit<4>
// MLIR: oq3.variable_assign @bitstring : !quir.cbit<4> = [[CAST6]]
cbit[4] bitstring = "0110";

// MLIR: [[CONST8:%.*]] = arith.constant 8 : i8
// MLIR: [[CAST8:%.*]] = "oq3.cast"([[CONST8]]) : (i8) -> !quir.cbit<8>
// MLIR: oq3.variable_assign @b : !quir.cbit<8> = [[CAST8]]
cbit[8] b = 8;

// MLIR: [[CONST5:%.*]] = arith.constant 5 : i4
// MLIR: [[CAST5:%.*]] = "oq3.cast"([[CONST5]]) : (i4) -> !quir.cbit<4>
// MLIR: oq3.variable_assign @c : !quir.cbit<4> = [[CAST5]]
cbit[4] c = 83957;

// initializer string shorter than the register
// Not yet supported: modeling the initializer value as a single 137-bit integer
// is conceptually fine, yet MLIR's asm printer code hits an assertion when
// turning the long integer into part of the value's name.
// MLIR: [[LONGBITREG_CONST:%.*]] = arith.constant 621124011108895393450852865781 : i137
// MLIR: [[LONGBITREG_CBIT:%.*]] = "oq3.cast"([[LONGBITREG_CONST]]) : (i137) -> !quir.cbit<137>
// MLIR: oq3.variable_assign @longbitreg : !quir.cbit<137> = [[LONGBITREG_CBIT]]
cbit[137] longbitreg = "0111110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101";
// compare python int("0b0111110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101", 0)

// initializer string is longer than the register
// As above
// MLIR: [[LONGBITREG2_CONST:%.*]] = arith.constant 73147070982778154320087907793426741712629 : i137
// MLIR: [[LONGBITREG2_CBIT:%.*]] = "oq3.cast"([[LONGBITREG2_CONST]]) : (i137) -> !quir.cbit<137>
// MLIR: oq3.variable_assign @longbitreg2 : !quir.cbit<137> = [[LONGBITREG2_CBIT]]
cbit[137] longbitreg2 = "10101101011011110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101";
// compare python int("0b<bitstring>", 0) & ((1<<137) - 1)

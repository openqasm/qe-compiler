OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.


// MLIR-DAG: quir.declare_variable @noninitialized : !quir.cbit<4>
// MLIR-DAG: quir.declare_variable @bitstring : !quir.cbit<4>
// MLIR-DAG: quir.declare_variable @b : !quir.cbit<8>
// MLIR-DAG: quir.declare_variable @c : !quir.cbit<4>


// MLIR: [[CONST0:%.*]] = arith.constant 0 : i4
// MLIR: [[CAST0:%.*]] = "quir.cast"([[CONST0]]) : (i4) -> !quir.cbit<4>
// MLIR: quir.assign_variable @noninitialized : !quir.cbit<4> = [[CAST0]]
cbit[4] noninitialized;

// MLIR: [[CONST6:%.*]] = arith.constant 6 : i4
// MLIR: [[CAST6:%.*]] = "quir.cast"([[CONST6]]) : (i4) -> !quir.cbit<4>
// MLIR: quir.assign_variable @bitstring : !quir.cbit<4> = [[CAST6]]
cbit[4] bitstring = "0110";

// MLIR: [[CONST8:%.*]] = arith.constant 8 : i8
// MLIR: [[CAST8:%.*]] = "quir.cast"([[CONST8]]) : (i8) -> !quir.cbit<8>
// MLIR: quir.assign_variable @b : !quir.cbit<8> = [[CAST8]]
cbit[8] b = 8;

// MLIR: [[CONST5:%.*]] = arith.constant 5 : i4
// MLIR: [[CAST5:%.*]] = "quir.cast"([[CONST5]]) : (i4) -> !quir.cbit<4>
// MLIR: quir.assign_variable @c : !quir.cbit<4> = [[CAST5]]
cbit[4] c = 83957;

// initializer string shorter than the register
// Not yet supported: modeling the initializer value as a single 137-bit integer
// is conceptually fine, yet MLIR's asm printer code hits an assertion when
// turning the long integer into part of the value's name.
// MLIR: [[LONGBITREG_CONST:%.*]] = arith.constant 621124011108895393450852865781 : i137
// MLIR: [[LONGBITREG_CBIT:%.*]] = "quir.cast"([[LONGBITREG_CONST]]) : (i137) -> !quir.cbit<137>
// MLIR: quir.assign_variable @longbitreg : !quir.cbit<137> = [[LONGBITREG_CBIT]]
cbit[137] longbitreg = "0111110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101";
// compare python int("0b0111110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101", 0)

// initializer string is longer than the register
// As above
// MLIR: [[LONGBITREG2_CONST:%.*]] = arith.constant 73147070982778154320087907793426741712629 : i137
// MLIR: [[LONGBITREG2_CBIT:%.*]] = "quir.cast"([[LONGBITREG2_CONST]]) : (i137) -> !quir.cbit<137>
// MLIR: quir.assign_variable @longbitreg2 : !quir.cbit<137> = [[LONGBITREG2_CBIT]]
cbit[137] longbitreg2 = "10101101011011110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101";
// compare python int("0b<bitstring>", 0) & ((1<<137) - 1)

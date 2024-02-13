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

// initializer strings are required by the spec to be the same size as the cbit
// Not yet supported: modeling the initializer value as a single 100-bit integer
// is conceptually fine, yet MLIR's asm printer code hits an assertion when
// turning the long integer into part of the value's name.
// MLIR: [[LONGBITREG_CONST:%.*]] = arith.constant 621124011108895393450852865781 : i100
// MLIR: [[LONGBITREG_CBIT:%.*]] = "oq3.cast"([[LONGBITREG_CONST]]) : (i100) -> !quir.cbit<100>
// MLIR: oq3.variable_assign @longbitreg : !quir.cbit<100> = [[LONGBITREG_CBIT]]
cbit[100] longbitreg = "0111110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101";
// compare python int("0b0111110101101111010110111101011011110101101111010110111101011011110101101111010110111101011011110101", 0)

OPENQASM 3;
// RUN: (! qss-compiler -X=qasm --emit=mlir --enable-parameters %s 2>&1 ) | FileCheck %s --check-prefixes NO-CIRCUITS,CIRCUITS
// RUN: (! qss-compiler -X=qasm --emit=mlir --enable-parameters --enable-circuits %s 2>&1 ) | FileCheck %s --check-prefix CIRCUITS 

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

// This test case validates error messages and warnings for input parameters

input angle theta = 3.141;
input float[64] theta2 = 1.56;

input complex badComplex;
input bit badBit;
input bool badBool;
input int badInt;
input uint badUInt;
input float[32] badFloat32;
input float badFloat;

// NO-CIRCUITS: loc("-":0:0): error: the --enable-parameters circuit requires --enable-circuits
// CIRCUITS-NOT: error: Input parameter theta type error. Input parameters must be angle or float[64].
// CIRCUITS-NOT: error: Input parameter theta2 type error. Input parameters must be angle or float[64].
// CIRCUITS-: error: Input parameter badComplex type error. Input parameters must be angle or float[64].
// CIRCUITS: error: Input parameter badBit type error. Input parameters must be angle or float[64].
// CIRCUITS: error: Input parameter badBool type error. Input parameters must be angle or float[64].
// CIRCUITS: error: Input parameter badInt type error. Input parameters must be angle or float[64].
// CIRCUITS: error: Input parameter badUInt type error. Input parameters must be angle or float[64].
OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s
//
// This test case validates OpenQASM 3's builtin constants pi, tau, and euler.

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


qubit $0;

U(pi, pi / 2, euler) $0;

angle[3] a = euler;

U(π, τ, ε) $0;

U(1.5 * π, τ / 2, 1.5 * ε) $0;


// gate declarations
gate phase(lambda) q {
  U(0, 0, lambda) q;
}

phase(1.8125 * pi) $0;
phase(1.8125 * π) $0;

phase(0.4 * tau) $0;
phase(0.4 * τ) $0;

phase(1.5 * euler) $0;
phase(1.5 * ε) $0;

angle phi = 1.5 * π + τ / 2 - 1.5 * ε;

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s
//
// This test case validates OpenQASM 3's builtin constants pi, tau, and euler.

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

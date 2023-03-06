OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s
//
// This test case validates OpenQASM 3's builtin constants pi, tau, and euler.

// (C) Copyright IBM 2023.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

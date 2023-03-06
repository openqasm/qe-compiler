OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST

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

// gate declarations
gate phase(lambda) q {
  U(0, 0, lambda) q;
}

gate cx c, t { }

gate h a {
  U(pi / 2, 0, pi) a;
}

qubit $0;
qubit $1;
angle[3] c = 0;

reset $0;
reset $1;
h $1;
h $0;


// AST: <StretchStatement>
// AST: <Stretch>
// AST: <Name>a</Name>
// AST: <Name>b</Name>
// AST: <Delay>
// AST: <DelayType>ASTTypeStretch</DelayType>
// AST: <Name>a</Name>

stretch a;
stretch b;
delay[a] $0;
delay[b] $1;
cx $0, $1;

// AST: <Op>*</Op>
phase(1.8125*pi) $1;
cx $0, $1;
phase(0.1875*pi) $1;
phase(0.1875*pi) $0;
h $0;
measure $0 -> c[0];
c <<= 1;

reset $0;
h $0;
stretch d;
delay[d] $1;
cx $0, $1;
phase(1.625*pi) $1;  // mod 2*pi
cx $0, $1;
phase(0.375*pi) $1;
angle[32] temp_1 = 0.375*pi;
temp_1 -= c;  // cast and do arithmetic mod 2 pi
phase(temp_1) $0;
h $0;
measure $0 -> c[0];

// AST: <Op><<=</Op>
c <<= 1;

reset $0;
h $0;
stretch f;
delay[f] $1;
cx $0, $1;
phase(1.25*pi) $1;  // mod 2*pi
cx $0, $1;
phase(0.75*pi) $1;
angle[32] temp_2 = 0.75*pi;
temp_2 -= c;  // cast and do arithmetic mod 2 pi
phase(temp_2) $0;
h $0;
measure $0 -> c[0];
c <<= 1;

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

//
// This code is part of Qiskit.
//
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

// MLIR: func @g(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) {
gate g qa, qb {
    // MLIR: quir.builtin_U %arg0{{.*}}
    // MLIR: quir.builtin_U %arg1{{.*}}
    U(1.57079632679, 0.0, 3.14159265359) qa;
    U(1.57079632679, 0.0, 3.14159265359) qb;
}

// MLIR: func @g4(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>, %arg2: !quir.qubit<1>) {
gate g4 qa, qb, qc {
    U(1.57079632679, 0.0, 3.14159265359) qa;
    U(0.0, 0.0, 3.14159265359) qb;
    U(1.57079632679, 0.0, 0.0) qc;
}

qubit $2;
qubit $3;
qubit $4;

// MLIR: quir.call_gate @g(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
// MLIR: quir.call_gate @g4(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()
g $2, $3;
g4 $2, $3, $4;

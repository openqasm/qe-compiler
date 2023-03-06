OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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

// MLIR: quir.builtin_U %arg0, {{.*}}, {{.*}}, %arg1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
gate g (theta) q {
    U(0.0, 0.0, theta) q;
}

// MLIR: quir.call_gate @g(%arg0, %arg1) : (!quir.qubit<1>, !quir.angle<64>) -> ()
gate q1 (theta) q {
    g(theta) q;
}

// MLIR: quir.call_gate @g(%arg0, %arg2) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// MLIR: quir.call_gate @g(%arg1, %arg3) : (!quir.qubit<1>, !quir.angle<64>) -> ()
gate g2 (theta, lambda) qa, qb {
    g(theta) qa;
    g(lambda) qb;
}

// qa = %arg0, qb = %arg1, theta = %arg2, lambda = %arg3, phi = %arg4
// MLIR: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR: quir.call_gate @g2(%arg0, %arg1, {{.*}}, {{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>) -> ()
// MLIR: {{.*}} = quir.constant #quir.angle<3.140000e+00 : !quir.angle<64>>
// MLIR: quir.call_gate @g2(%arg1, %arg0, {{.*}}, %arg2) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>) -> ()
// MLIR: quir.call_gate @g2(%arg0, %arg1, %arg4, %arg3) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>) -> ()
gate g3 (theta, lambda, phi) qa, qb {
    g2(0.0, 0.0) qa, qb;
    g2(3.14, theta) qb, qa;
    g2(phi, lambda) qa, qb;
}

// MLIR: func @main() -> i32 {

// MLIR: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// MLIR: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
qubit $2;
qubit $3;

// MLIR: quir.call_gate @g([[QUBIT2]], %{{.*}}) : (!quir.qubit<1>, !quir.angle<64>) -> ()
// MLIR: quir.call_gate @g2([[QUBIT2]], [[QUBIT3]], %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>) -> ()
g (3.14) $2;
g2 (3.14, 1.2) $2, $3;
g3 (3.14, 1.2, 0.2) $2, $3;

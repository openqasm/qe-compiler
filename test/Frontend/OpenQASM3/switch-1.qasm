OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
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

int i = 15;
qubit $0;
// MLIR: quir.switch %{{.*}}{
// MLIR:     %angle = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR:     %angle_0 = quir.constant #quir.angle<1.000000e-01 : !quir.angle<64>>
// MLIR:     %angle_1 = quir.constant #quir.angle<2.000000e-01 : !quir.angle<64>>
// MLIR:     quir.builtin_U %{{.*}}, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: }[1 : {
// MLIR: }2 : {
// MLIR: }3 : {
// MLIR: }5 : {
// MLIR: }12 : {
// MLIR: }]
// AST-PRETTY: SwitchStatementNode(SwitchQuantity(name=i, type=ASTTypeIdentifier),
switch (i) {
    // AST-PRETTY: statements=[
    // AST-PRETTY: CaseStatementNode(case=1, ),
    case 1: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=2, ),
    case 2: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=3, ),
    case 3: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=5, ),
    case 5: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=12, ),
    case 12: {
    }
    break;
    // AST-PRETTY: ],
    // AST-PRETTY: default statement=[
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.1, bits=64), AngleNode(value=0.2, bits=64)], qubits=[], qcparams=[$0])
    // AST-PRETTY: ])
    default: {
        U(0.0, 0.1, 0.2) $0;
    }
    break;
}

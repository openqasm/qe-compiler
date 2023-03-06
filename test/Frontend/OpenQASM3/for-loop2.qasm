OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: (qss-compiler -o /dev/null -X=qasm --emit=mlir %s || true) 2>&1 | FileCheck %s --check-prefix MLIR-DIAGNOSTICS

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

// This test serves to validate parsing and walking of the AST for for-loop
// features that are currently unsupported.
// NOTE: This test also validates diagnostic messages emitted by QUIRGen. When
//       implementing the for-loop features, introduce a replacement test for
//       the diagnostics.

// AST-PRETTY: DeclarationNode(type=ASTTypeQubitContainer, QubitContainerNode(QubitNode(name=$0:0, bits=1)))
qubit $0;

// MLIR-DIAGNOSTICS: error: Negative stepping values in a for loop are not yet supported in QUIRGen.
// AST-PRETTY: ForStatementNode(start=10, stepping=-1, end=1,
for i in [10 : -1 : 1] {
    // AST-PRETTY: statements=
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=1.57079632679000003037, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=3.14159265359000006157, bits=64)], qubits=[], qcparams=[$0])

    // AST-PRETTY: )
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

// MLIR-DIAGNOSTICS: error: Discrete values in a for loop are not yet supported in QUIRGen.
// AST-PRETTY: ForStatementNode(i=[0 1 4 ], stepping=-1,
for i in { 0, 1, 4 } {
    U(1.57079632679, 0.0, 3.14159265359) $0;
}

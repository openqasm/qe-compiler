OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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

// MLIR: %{{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// AST-PRETTY: DeclarationNode(type=ASTTypeAngle, AngleNode(value=0.0, bits=64))
angle[64] angle_a;

// MLIR: %{{.*}} = quir.constant #quir.angle<0.70699977874755859> : !quir.angle<20>
// MLIR: %{{.*}} = quir.constant #quir.angle<0.69999980926513672> : !quir.angle<20>
// AST-PRETTY: DeclarationNode(type=ASTTypeAngle, AngleNode(value=0.70699978, bits=20))
// AST-PRETTY: DeclarationNode(type=ASTTypeAngle, AngleNode(value=0.69999981, bits=20))
angle[20] my_angle = 0.707;
angle[20] second_angle = 0.7;

// The remainder of this test case will be re-activated as part of IBM-Q-Software/QSS-Compiler#220
// COM: MLIR: %{{.*}} = cmpi eq, %{{.*}}, %{{.*}} : !quir.angle<20>
// COM: AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=IdentifierNode(name=my_angle, bits=20), right=IdentifierNode(name=second_angle, bits=20)),
qubit $0;
// if (my_angle == second_angle) {
//
//   // COM: MLIR: %{{.*}} = quir.constant #quir.angle<5.000000e-01> : !quir.angle<64>
//   // COM: MLIR: %{{.*}} = quir.constant #quir.angle<0.000000e+00> : !quir.angle<32>
//   // COM: MLIR: %{{.*}} = quir.constant #quir.angle<0.32432432> : !quir.angle<64>
//   U(0.5, 0, 0.32432432) $0;
//
// }

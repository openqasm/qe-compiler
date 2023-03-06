OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s | FileCheck %s --check-prefix AST
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
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

// MLIR: qcs.init
// MLIR: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
qubit $0;

// AST: <StatementList>
// AST: <ResetNode>
// AST: <GateQOpNode>
// AST: <GateOpNode>
// AST: </GateOpNode>
// AST: </GateQOpNode>
// AST: <TargetName>$0</TargetName>
// AST: </ResetNode>
// AST: </StatementList>
// AST-PRETTY: ResetNode(IdentifierNode(name=$0, bits=1))
// MLIR: quir.reset [[QUBIT0]] : !quir.qubit<1>
reset $0;

// MLIR: qcs.finalize

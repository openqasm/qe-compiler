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

// MLIR-DAG: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// MLIR-DAG: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
qubit $2;
qubit $3;

// MLIR-DAG: quir.declare_variable @is_excited : !quir.cbit<1>
// MLIR-DAG: quir.declare_variable @other : !quir.cbit<1>
// MLIR-DAG: quir.declare_variable @result : !quir.cbit<1>
bit is_excited;
bit other;
bit result;

gate x q {
   // TODO re-enable as part of IBM-Q-Software/qss-compiler#586
   // U(pi, 0, pi) q;
}

x $2;
x $3;

// MLIR: [[MEASURE2:%.*]] = quir.measure([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @is_excited<1> [0] : i1 = [[MEASURE2]]
is_excited = measure $2;

// Apply reset operation

// MLIR: [[EXCITED:%.*]] = quir.use_variable @is_excited : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[EXCITEDCAST:%[0-9]+]] = "quir.cast"([[EXCITED]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND0:%.*]] = arith.cmpi eq, [[EXCITEDCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND0]] {
if (is_excited == 1) {
// MLIR: [[MEASURE3:%.*]] = quir.measure([[QUBIT3]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @other<1> [0] : i1 = [[MEASURE3]]
  other = measure $3;
// MLIR: [[OTHER:%.*]] = quir.use_variable @other : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[OTHERCAST:%[0-9]+]] = "quir.cast"([[OTHER]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND1:%.*]] = arith.cmpi eq, [[OTHERCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND1]] {
  if (other == 1){
// MLIR: quir.call_gate @x([[QUBIT2]]) : (!quir.qubit<1>) -> ()
     x $2;
  }
}
// MLIR: [[MEASURE2:%.*]] = quir.measure([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// MLIR: quir.assign_cbit_bit @result<1> [0] : i1 = [[MEASURE2]]
result = measure $2;

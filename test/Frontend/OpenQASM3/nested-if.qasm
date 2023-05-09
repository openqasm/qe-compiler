OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

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

// MLIR-DAG: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// MLIR-DAG: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
qubit $2;
qubit $3;

// MLIR-DAG: oq3.declare_variable @is_excited : !quir.cbit<1>
// MLIR-DAG: oq3.declare_variable @other : !quir.cbit<1>
// MLIR-DAG: oq3.declare_variable @result : !quir.cbit<1>
bit is_excited;
bit other;
bit result;

gate x q {
   // TODO re-enable as part of IBM-Q-Software/qss-compiler#586
   // U(pi, 0, pi) q;
}

x $2;
x $3;

// MLIR-NO-CIRCUITS: [[MEASURE2:%.*]] = quir.measure([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE2:%.*]] = quir.call_circuit @circuit_0([[QUBIT3]], [[QUBIT2]]) : (!quir.qubit<1>, !quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @is_excited<1> [0] : i1 = [[MEASURE2]]
is_excited = measure $2;

// Apply reset operation

// MLIR: [[EXCITED:%.*]] = oq3.variable_load @is_excited : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[EXCITEDCAST:%[0-9]+]] = "oq3.cast"([[EXCITED]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND0:%.*]] = arith.cmpi eq, [[EXCITEDCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND0]] {
if (is_excited == 1) {
// MLIR-NO-CIRCUITS: [[MEASURE3:%.*]] = quir.measure([[QUBIT3]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE3:%.*]] = quir.call_circuit @circuit_1([[QUBIT3]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @other<1> [0] : i1 = [[MEASURE3]]
  other = measure $3;
// MLIR: [[OTHER:%.*]] = oq3.variable_load @other : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[OTHERCAST:%[0-9]+]] = "oq3.cast"([[OTHER]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND1:%.*]] = arith.cmpi eq, [[OTHERCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND1]] {
  if (other == 1){
// MLIR-NO-CIRCUITS: quir.call_gate @x([[QUBIT2]]) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS: quir.call_circuit @circuit_2([[QUBIT2]]) : (!quir.qubit<1>) -> ()
     x $2;
  }
}
// MLIR-NO-CIRCUITS: [[MEASURE2:%.*]] = quir.measure([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// MLIR-CIRCUITS: [[MEASURE2:%.*]] = quir.call_circuit @circuit_3([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @result<1> [0] : i1 = [[MEASURE2]]
result = measure $2;

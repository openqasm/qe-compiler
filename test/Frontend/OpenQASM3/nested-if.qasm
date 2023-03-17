OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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

// MLIR: [[MEASURE2:%.*]] = quir.measure([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @is_excited<1> [0] : i1 = [[MEASURE2]]
is_excited = measure $2;

// Apply reset operation

// MLIR: [[EXCITED:%.*]] = oq3.variable_load @is_excited : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[EXCITEDCAST:%[0-9]+]] = "oq3.cast"([[EXCITED]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND0:%.*]] = arith.cmpi eq, [[EXCITEDCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND0]] {
if (is_excited == 1) {
// MLIR: [[MEASURE3:%.*]] = quir.measure([[QUBIT3]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @other<1> [0] : i1 = [[MEASURE3]]
  other = measure $3;
// MLIR: [[OTHER:%.*]] = oq3.variable_load @other : !quir.cbit<1>
// MLIR: [[CONST:%[0-9a-z_]+]] = arith.constant 1 : i32
// MLIR: [[OTHERCAST:%[0-9]+]] = "oq3.cast"([[OTHER]]) : (!quir.cbit<1>) -> i32
// MLIR: [[COND1:%.*]] = arith.cmpi eq, [[OTHERCAST]], [[CONST]] : i32
// MLIR: scf.if [[COND1]] {
  if (other == 1){
// MLIR: quir.call_gate @x([[QUBIT2]]) : (!quir.qubit<1>) -> ()
     x $2;
  }
}
// MLIR: [[MEASURE2:%.*]] = quir.measure([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// MLIR: oq3.cbit_assign_bit @result<1> [0] : i1 = [[MEASURE2]]
result = measure $2;

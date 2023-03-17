// RUN: qss-compiler -X=mlir --quir-eliminate-variables %s --canonicalize| FileCheck %s --implicit-check-not '!quir.cbit' --implicit-check-not variable --implicit-check-not alloc --implicit-check-not store
//
// This test verifies store-forwarding for multi-bit registers.

// CHECK: module
module {
  oq3.declare_variable @a : !quir.cbit<1>
  oq3.declare_variable @b : !quir.cbit<2>
  func @x(%arg0: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c0_i2 = arith.constant 0 : i2
    %true = arith.constant true
    qcs.init
    qcs.shot_init {qcs.num_shots = 1 : i32}

    // CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0
    // CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1
    // CHECK: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2
    %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>

    %3 = "oq3.cast"(%true) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @a : !quir.cbit<1> = %3
    %4 = "oq3.cast"(%c0_i2) : (i2) -> !quir.cbit<2>
    oq3.variable_assign @b : !quir.cbit<2> = %4

    // CHECK: [[MEASURE0:%.*]] = quir.measure([[QUBIT0]])
    %5 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    oq3.cbit_assign_bit @b<2> [0] : i1 = %5

    // CHECK: [[MEASURE1:%.*]] = quir.measure([[QUBIT1]])
    %6 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    oq3.cbit_assign_bit @b<2> [1] : i1 = %6

    // CHECK: [[MEASURE2:%.*]] = quir.measure([[QUBIT2]])
    %7 = quir.measure(%2) : (!quir.qubit<1>) -> i1
    %8 = "oq3.cast"(%7) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @a : !quir.cbit<1> = %8

    %9 = oq3.variable_load @b : !quir.cbit<2>
    %10 = oq3.cbit_extractbit(%9 : !quir.cbit<2>) [0] : i1

    // CHECK: scf.if [[MEASURE0]]
    scf.if %10 {
      quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
    }

    %11 = oq3.variable_load @b : !quir.cbit<2>
    %12 = oq3.cbit_extractbit(%11 : !quir.cbit<2>) [1] : i1

    // CHECK: scf.if [[MEASURE1]]
    scf.if %12 {
      quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
    }

    %13 = oq3.variable_load @a : !quir.cbit<1>
    %14 = "oq3.cast"(%13) : (!quir.cbit<1>) -> i1

    // CHECK: scf.if [[MEASURE2]]
    scf.if %14 {
      quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
    }
    qcs.finalize
    return %c0_i32 : i32
  }
}

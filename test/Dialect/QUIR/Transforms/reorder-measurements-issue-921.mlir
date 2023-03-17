// RUN: qss-compiler --canonicalize --quantum-decorate --reorder-measures %s | FileCheck %s

// This regression test case validates that reorder-measures can move
// operations across control flow when there are no conflicts, and also that
// conflicts prevent movement

// based on failures for
// reset $2;
// cx $1, $4;
// reset $1;
//
// This caused issues with measurement reordering being less smart than the
// waveform scheduler, which corrupted the metadata and caused localization to
// fail.

// CHECK: module
module {
  oq3.declare_variable @a : !quir.cbit<1>
  oq3.declare_variable @b : !quir.cbit<1>
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    qcs.shot_init {qcs.num_shots = 1000 : i32}
// CHECK:  [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
// CHECK:  [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
// CHECK:  [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
    %0 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    %1 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    %2 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
// CHECK: quir.builtin_CX [[QUBIT1]], [[QUBIT3]] : !quir.qubit<1>, !quir.qubit<1>
// CHECK: %{{.*}} = quir.measure([[QUBIT2]]) : (!quir.qubit<1>) -> i1
// CHECK: quir.call_gate @rz([[QUBIT1]], {{.*}})
// CHECK: %{{.*}} = quir.measure([[QUBIT1]]) : (!quir.qubit<1>) -> i1
    %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
    %6 = "oq3.cast"(%5) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @a : !quir.cbit<1> = %6
    %7 = oq3.variable_load @a : !quir.cbit<1>
    %8 = "oq3.cast"(%7) : (!quir.cbit<1>) -> i32
    %9 = arith.cmpi eq, %8, %c1_i32 : i32
    scf.if %9 {
      quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
    } {quir.classicalOnly = false, quir.physicalIds = [2 : i32]}
    quir.builtin_CX %0, %2 : !quir.qubit<1>, !quir.qubit<1>
    %angle = "oq3.cast"(%5) : (i1) -> !quir.angle<64>
    // The following gate call uses a value generated from the measurement
    quir.call_gate @rz(%0, %angle) : (!quir.qubit<1>, !quir.angle<64>) -> ()
    %10 = quir.measure(%0) : (!quir.qubit<1>) -> i1
    %11 = "oq3.cast"(%10) : (i1) -> !quir.cbit<1>
    oq3.variable_assign @b : !quir.cbit<1> = %11
    %12 = oq3.variable_load @b : !quir.cbit<1>
    %13 = "oq3.cast"(%12) : (!quir.cbit<1>) -> i32
    %14 = arith.cmpi eq, %13, %c1_i32 : i32
    scf.if %14 {
      quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
    } {quir.classicalOnly = false, quir.physicalIds = [1 : i32]}
    qcs.finalize
    return %c0_i32 : i32
  }
}

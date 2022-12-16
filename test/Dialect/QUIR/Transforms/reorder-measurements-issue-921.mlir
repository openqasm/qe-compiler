// RUN: qss-compiler --canonicalize --quantum-decorate --reorder-measures %s | FileCheck %s

// This regression test case validates that reorder-measures can move
// operations across control flow when there are no conflicts

// Related to https://github.ibm.com/IBM-Q-Software/ibm-qss-compiler/issues/921

// based on failures for 
// reset $2;
// cx $1, $4;
// reset $1;

// CHECK: module
module {
  quir.declare_variable @a : !quir.cbit<1>
  quir.declare_variable @b : !quir.cbit<1>
  func @rz(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) attributes {quir.classicalOnly = false} {
    return
  }
  func @sx(%arg0: !quir.qubit<1>) attributes {quir.classicalOnly = false} {
    return
  }
  func @x(%arg0: !quir.qubit<1>) attributes {quir.classicalOnly = false} {
    return
  }
  func @id(%arg0: !quir.qubit<1>) attributes {quir.classicalOnly = false} {
    return
  }
  func @cx(%arg0: !quir.qubit<1>, %arg1: !quir.qubit<1>) attributes {quir.classicalOnly = false} {
    return
  }
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %dur = quir.constant #quir.duration<"1ms" : !quir.duration>
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    quir.system_init
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      quir.delay %dur, () : !quir.duration, () -> ()
      quir.shot_init {quir.numShots = 1000 : i32}
      %0 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      %1 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      %2 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
      %3 = "quir.cast"(%false) : (i1) -> !quir.cbit<1>
      quir.assign_variable @a : !quir.cbit<1> = %3
      %4 = "quir.cast"(%false) : (i1) -> !quir.cbit<1>
      quir.assign_variable @b : !quir.cbit<1> = %4
      %5 = quir.measure(%1) : (!quir.qubit<1>) -> i1
      %6 = "quir.cast"(%5) : (i1) -> !quir.cbit<1>
      quir.assign_variable @a : !quir.cbit<1> = %6
      %7 = quir.use_variable @a : !quir.cbit<1>
      %8 = "quir.cast"(%7) : (!quir.cbit<1>) -> i32
      %9 = arith.cmpi eq, %8, %c1_i32 : i32
      scf.if %9 {
        quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
      } {quir.classicalOnly = false, quir.physicalIds = [2 : i32]}
      quir.builtin_CX %0, %2 : !quir.qubit<1>, !quir.qubit<1>
      %10 = quir.measure(%0) : (!quir.qubit<1>) -> i1
      %11 = "quir.cast"(%10) : (i1) -> !quir.cbit<1>
      quir.assign_variable @b : !quir.cbit<1> = %11
      %12 = quir.use_variable @b : !quir.cbit<1>
      %13 = "quir.cast"(%12) : (!quir.cbit<1>) -> i32
      %14 = arith.cmpi eq, %13, %c1_i32 : i32
      scf.if %14 {
        quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
      } {quir.classicalOnly = false, quir.physicalIds = [1 : i32]}
    } {quir.classicalOnly = false, quir.physicalIds = [1 : i32, 2 : i32, 3 : i32], quir.shotLoop}
    quir.system_finalize
    return %c0_i32 : i32
  }
}

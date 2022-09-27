// RUN: qss-compiler -X=mlir %s --quantum-decorate | FileCheck %s

func @t1 (%cond : i1) -> () {
  %q0 = quir.declare_qubit {id = 0: i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1: i32} : !quir.qubit<1>
  %q2 = quir.declare_qubit {id = 2: i32} : !quir.qubit<1>
  %q3 = quir.declare_qubit {id = 3: i32} : !quir.qubit<1>
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32]}
  }
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32]}
  }
  scf.if %cond {
    quir.call_gate @x(%q0) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32, 2 : i32, 3 : i32]}
  }
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    %res = "quir.measure"(%q1) : (!quir.qubit<1>) -> i1
    quir.reset %q0 : !quir.qubit<1>
    quir.call_gate @x(%q3) : (!quir.qubit<1>) -> ()
    quir.call_gate @x(%q2) : (!quir.qubit<1>) -> ()
    // CHECK: {quir.physicalIds = [0 : i32, 1 : i32, 2 : i32, 3 : i32]}
  }
  return
}

// RUN: qss-compiler -X=mlir --merge-resets-lexicographic -emit=mlir %s | FileCheck %s
// RUN: qss-compiler -X=mlir --merge-resets-topological -emit=mlir %s | FileCheck %s --check-prefix=TOPO
//
// This test case checks that the MergeResetPass merges all parallelizable
// resets (and no more than that).

module  {
  func @x(%arg0: !quir.qubit<1>) {
    return
  }
  func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      %0 = oq3.declare_duration {value = "1ms"} : !quir.duration
      quir.delay %0, () : !quir.duration, () -> ()

      // qubit $0;
      // qubit $1;
      // qubit $2;
      // qubit $3;
      %1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      // CHECK: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      // TOPO: [[QUBIT0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      %2 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      // CHECK: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      // TOPO: [[QUBIT1:%.*]] = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      %3 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      // CHECK: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      // TOPO: [[QUBIT2:%.*]] = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      %4 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
      // CHECK: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
      // TOPO: [[QUBIT3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>

      // Expect that parallelizable resets get merged
      // reset $0;
      // reset $1;
      // reset $2;
      // reset $3;
      quir.reset %1 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT0]] : !quir.qubit<1>
      quir.reset %2 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT1]] : !quir.qubit<1>
      quir.reset %3 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT2]] : !quir.qubit<1>
      quir.reset %4 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT3]] : !quir.qubit<1>
      // CHECK: quir.reset [[QUBIT0]], [[QUBIT1]], [[QUBIT2]], [[QUBIT3]] : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>
      // TOPO: quir.reset [[QUBIT0]], [[QUBIT1]], [[QUBIT2]], [[QUBIT3]] : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>


      // Expect that resets are not merged across barriers
      // CHECK: quir.barrier
      // barrier $0, $1, $2, $3;
      // reset $0;
      // reset $1;
      // barrier $2, $3;
      // reset $2;
      // reset $3;
      quir.barrier %1, %2, %3, %4 : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()
      // CHECK: quir.reset [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>
      // TOPO: quir.reset [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>
      quir.reset %1 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT0]] : !quir.qubit<1>
      quir.reset %2 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT1]] : !quir.qubit<1>
      quir.barrier %3, %4 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
      // CHECK: quir.reset [[QUBIT2]], [[QUBIT3]] : !quir.qubit<1>, !quir.qubit<1>
      // TOPO: quir.reset [[QUBIT2]], [[QUBIT3]] : !quir.qubit<1>, !quir.qubit<1>
      quir.reset %3 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT2]] : !quir.qubit<1>
      quir.reset %4 : !quir.qubit<1>
      // CHECK-NOT: quir.reset [[QUBIT3]] : !quir.qubit<1>

      // CHECK-NOT: quir.reset {{.*}} : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>

      quir.barrier %1, %2 : (!quir.qubit<1>, !quir.qubit<1>) -> ()

      // Check that resets are not merged across gates
      // gate x q { }
      // CHECK: quir.barrier
      // barrier $0, $1;
      // reset $0;
      // x $3;
      // reset $1;
      quir.reset %1 : !quir.qubit<1>
      // CHECK: quir.reset [[QUBIT0]] : !quir.qubit<1>
      quir.call_gate @x(%4) : (!quir.qubit<1>) -> ()
      %cst = constant unit
      quir.reset %2 : !quir.qubit<1>
      // CHECK: quir.reset [[QUBIT1]] : !quir.qubit<1>
      // TOPO: quir.reset [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>

      // Check that resets with overlapping qubits do not get merged
      // barrier $0, $1;
      // reset $0;
      // reset $1;
      // reset $2;
      // reset $0;
      // reset $1;

      quir.barrier %1, %2 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
      // CHECK-NOT: quir.reset {{.*}} : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>
      // TOPO-NOT: quir.reset {{.*}} : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>
      quir.reset %1 : !quir.qubit<1>
      quir.reset %2 : !quir.qubit<1>
      // CHECK: quir.reset [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>
      // TOPO: quir.reset [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>
      quir.reset %3 : !quir.qubit<1>
      quir.reset %1 : !quir.qubit<1>
      quir.reset %2 : !quir.qubit<1>
      // CHECK: quir.reset [[QUBIT2]], [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>
      // TOPO: quir.reset [[QUBIT2]], [[QUBIT0]], [[QUBIT1]] : !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>

      quir.barrier %1, %2, %3, %4 : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()
      // TOPO: quir.barrier [[QUBIT0]], [[QUBIT1]], [[QUBIT2]], [[QUBIT3]] : (!quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>, !quir.qubit<1>) -> ()

      quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
      quir.reset %1 : !quir.qubit<1>
      quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()

      // Ideally we would be able to merge these sequences to become:
      // x $1; x $2; reset $1, $2; x $1, x $2
      // But topological merging is not enough to do that
      // TODO: Scheduler aware merging and uncomment the following line?
      //quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
      quir.reset %2 : !quir.qubit<1>
      quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()

      // TOPO: quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
      // TOPO: quir.reset %1, %2 : !quir.qubit<1>, !quir.qubit<1>
      // TOPO: quir.call_gate @x(%1) : (!quir.qubit<1>) -> ()
      // TOPO: quir.call_gate @x(%2) : (!quir.qubit<1>) -> ()
    }
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

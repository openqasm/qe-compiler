// RUN: qss-compiler -X=mlir --enable-circuits=true --extract-circuits %s | FileCheck %s
module {
  oq3.declare_variable @obs : !quir.cbit<4>
  func.func @x(%arg0: !quir.qubit<1>) attributes {quir.classicalOnly = false} {
    return
  }
  // CHECK: quir.circuit @circuit_0
  // CHECK: quir.delay %arg0, (%arg1)
  // CHECK: %0:2 = quir.measure(%arg2, %arg3)
  // CHECK: quir.return %0#0, %0#1 : i1, i1
  // CHECK: quir.circuit @circuit_1
  // CHECK: quir.call_gate @x(%arg0)
  // CHECK: quir.return
  // CHECK: func.func @main()
  func.func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i4 = arith.constant 0 : i4
    %dur = quir.constant #quir.duration<1.000000e+00> : !quir.duration<ms>
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %dur_0 = quir.constant #quir.duration<2.800000e+03> : !quir.duration<dt>
    qcs.init
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      quir.delay %dur, () : !quir.duration<ms>, () -> ()
      qcs.shot_init {qcs.num_shots = 1000 : i32}
      %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      %1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
      %2 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
      %3 = "oq3.cast"(%c0_i4) : (i4) -> !quir.cbit<4>
      oq3.variable_assign @obs : !quir.cbit<4> = %3
      quir.delay %dur_0, (%1) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
      %4:2 = quir.measure(%0, %2) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK-NOT: quir.delay %dur_0, (%1) : !quir.duration<dt>, (!quir.qubit<1>) -> ()
      // CHECK-NOT: %4:2 = quir.measure(%0, %2) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
      // CHECK: %4:2 = quir.call_circuit @circuit_0(%dur_0, %1, %0, %2)
      scf.if %4#0 {
        quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
        // CHECK-NOT:  quir.call_gate @x(%0) : (!quir.qubit<1>) -> ()
        // CHECK: quir.call_circuit @circuit_1(%0) : (!quir.qubit<1>) -> ()
      } {quir.classicalOnly = false, quir.physicalIds = [0 : i32]}
      oq3.cbit_assign_bit @obs<4> [0] : i1 = %4#1
    } {qcs.shot_loop, quir.classicalOnly = false, quir.physicalIds = [0 : i32, 1 : i32, 2 : i32]}
    qcs.finalize
    return %c0_i32 : i32
  }
}

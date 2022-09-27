// RUN: qss-compiler -X=mlir --convert-quir-angles %s | FileCheck %s
module  {
  func @rz(%arg0: !quir.qubit<1>, %arg1: !quir.angle<64>) {
    return
  }
  func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = quir.declare_qubit {id = 2 : i32} : !quir.qubit<1>
    // CHECK: {{.*}} = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
    %1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<32>>
    quir.call_gate @rz(%0, %1) : (!quir.qubit<1>, !quir.angle<32>) -> ()
    quir.system_finalize
    return %c0_i32 : i32
  }
}

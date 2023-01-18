// RUN: qss-compiler %s --config %TEST_CFG --target mock --canonicalize --mock-quir-to-std --emit=qem --plaintext-payload  | FileCheck %s

// CHECK: define i32 @main() !dbg !3 {
// CHECK:  ret i32 0, !dbg !7
// CHECK: }
module @controller attributes {quir.nodeId = 1000 : i32, quir.nodeType = "controller"}  {
  func @main() -> i32 attributes {quir.classicalOnly = false} {
    %0 = quir.declare_duration {value = "1000dt"} : !quir.duration
    %1 = sys.recv {fromId = 0 : index} : i1
    sys.broadcast %1 : i1
    scf.if %1 {
    } {quir.classicalOnly = false}
    %2 = sys.recv {fromId = 0 : index} : i1
    sys.broadcast %2 : i1
    scf.if %2 {
    } {quir.classicalOnly = false}
    %3 = sys.recv {fromId = 0 : index} : i1
    sys.broadcast %3 : i1
    scf.if %3 {
    } {quir.classicalOnly = false}
    %4 = llvm.mlir.constant(1.0) : i32
    %5 = llvm.mlir.constant(2.0) : i32
    %6 = "llvm.intr.pow"(%4, %5) : (i32, i32) -> i32
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

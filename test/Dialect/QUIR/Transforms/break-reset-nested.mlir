// RUN: qss-compiler -X=mlir --break-reset %s | FileCheck %s

// CHECK: scf.if %arg1 {
// CHECK-NEXT: scf.if %arg2 {
// CHECK-NEXT:   %{{.*}} = quir.measure(%arg0) {quir.noReportOuroboros} : (!quir.qubit<1>) -> i1
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     quir.call_gate @x(%arg0) : (!quir.qubit<1>) -> ()
func @main (%inq : !quir.qubit<1>, %cond1 : i1, %cond2 : i1) {
  scf.if %cond1 {
    scf.if %cond2 {
      quir.reset %inq : !quir.qubit<1>
    }
  }
  return
}

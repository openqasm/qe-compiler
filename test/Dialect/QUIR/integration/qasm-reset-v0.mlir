// RUN: qss-compiler -X=mlir %s | FileCheck %s
// CHECK: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
%q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
// CHECK: quir.reset %{{.*}} : !quir.qubit<1>
quir.reset %q0 : !quir.qubit<1>

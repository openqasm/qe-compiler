OPENQASM 3.0;
// RUN: qss-compiler --num-shots=1  %s | FileCheck %s
//
// Test implicit bit to int cast in comparisons.

qubit $0;

bit[5] a = "10101";

gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}

x $0;

// Test implicit cast of bit[n] to int
// CHECK:       %{{.*}} = arith.constant 21 : i32
// CHECK-NEXT:  %{{.*}} = "quir.cast"(%{{.*}}) : (!quir.cbit<5>) -> i32
// CHECK-NEXT:  %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
if(a == 21){
	x $0;
}

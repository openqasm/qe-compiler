// RUN: qss-compiler %s | FileCheck %s
// Verify that a test with no classical ops inside a circuit passes.

// CHECK-LABEL: quir.circuit @circuit1(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.angle<32>) -> i1 {
quir.circuit @circuit1 (%q0 : !quir.qubit<1>, %q1 : !quir.qubit<1>, %omega: !quir.angle<32>) -> i1 {
	// CHECK: quir.builtin_CX %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.qubit<1>
	quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
	// CHECK: quir.call_gate @rx(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	quir.call_gate @rx(%q0, %omega) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	// CHECK: %{{.*}} = quir.measure(%{{.*}})
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	// CHECK: quir.return %{{.*}}
	quir.return %res0 : i1
}

// CHECK-LABEL: quir.circuit @circuit2(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.angle<32>) -> i1 {
quir.circuit @circuit2 (%q0 : !quir.qubit<1>, %q1 : !quir.qubit<1>, %omega: !quir.angle<32>) -> i1 {
	// CHECK: quir.reset %{{.*}} : !quir.qubit<1>
	quir.reset %q0 : !quir.qubit<1>
	// CHECK: quir.barrier %{{.*}}, %{{.*}} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
	quir.barrier %q0, %q1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
	// CHECK: quir.builtin_CX %{{.*}}, %{{.*}} : !quir.qubit<1>, !quir.qubit<1>
	quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
	// CHECK: quir.call_gate @h(%{{.*}}) : (!quir.qubit<1>) -> ()
	quir.call_gate @h(%q0) : (!quir.qubit<1>) -> ()
	// CHECK: quir.call_gate @rx(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	quir.call_gate @rx(%q1, %omega) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	// CHECK: quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
	quir.call_gate @x(%q1) : (!quir.qubit<1>) -> ()
	// CHECK: %{{.*}} = quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	// CHECK: quir.return %{{.*}}
	quir.return %res0 : i1
}

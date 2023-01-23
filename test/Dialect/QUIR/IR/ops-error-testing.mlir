// RUN: qss-opt %s -split-input-file -verify-diagnostics

// -----

quir.circuit @circuit1 (%q0 : !quir.qubit<1>, %omega1: !quir.angle<32>, %omega2: !quir.angle<32>) -> i1 {
	// expected-error@+1 {{'quir.angle_add' op is classical and should not be inside a circuit.}}
	%omega3 = quir.angle_add %omega1, %omega2 : !quir.angle<32>
	quir.call_gate @rx(%q0, %omega3) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0 : i1
}

// -----

quir.circuit @circuit2 (%q0 : !quir.qubit<1>, %omega1: !quir.angle<32>) -> i1 {
	quir.call_gate @rx(%q0, %omega1) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	// expected-error@+1 {{'quir.cast' op is classical and should not be inside a circuit.}}
	%f1 = "quir.cast"(%omega1) : (!quir.angle<32>) -> f64
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0 : i1
}

// -----

quir.circuit @circuit3 (%q0 : !quir.qubit<1>, %omega1: !quir.angle<32>) -> i1 {
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	// expected-error@+1 {{'scf.yield' op is classical and should not be inside a circuit.}}
	scf.if %res0 {
		quir.call_gate @rx(%q0, %omega1) : (!quir.qubit<1>, !quir.angle<32>) -> ()
	}
	quir.return %res0 : i1
}

// -----

quir.circuit @circuit4 (%q0 : !quir.qubit<1>, %omega1: !quir.angle<32>) -> i1 {
	// expected-error@+1 {{'oq3.declare_variable' op is classical and should not be inside a circuit.}}
	oq3.declare_variable @c1 : !quir.cbit<1>
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0 : i1
}

// -----

quir.circuit @circuit5 (%q0 : !quir.qubit<1>, %q1 : !quir.qubit<1>, %omega1: !quir.angle<32>) -> i1 {
	%ub = arith.constant 10 : index
	%cmpval = arith.constant 9 : index
	// expected-error@+1 {{'arith.cmpi' op is classical and should not be inside a circuit.}}
	%cond = arith.cmpi "eq", %ub, %cmpval : index
	quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0 : i1
}

// -----

quir.circuit @circuit6 (%q0 : !quir.qubit<1>, %omega1: !quir.angle<32>, %omega2: !quir.angle<32>) -> i1 {
	// expected-error@+1 {{'quir.angle_cmp' op is classical and should not be inside a circuit.}}
	%b0 = quir.angle_cmp {predicate = "eq"} %omega1, %omega2 : !quir.angle<32> -> i1
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0 : i1
}

// -----

quir.circuit @circuit7 (%q0 : !quir.qubit<1>, %omega1: !quir.angle<32>, %omega2: !quir.angle<32>) -> i1 {
	// expected-error@+1 {{'quir.declare_duration' op is classical and should not be inside a circuit.}}
	%l1 = quir.declare_duration {value = "10ns"} : !quir.duration
	%res0 = "quir.measure"(%q0) : (!quir.qubit<1>) -> i1
	quir.return %res0 : i1
}

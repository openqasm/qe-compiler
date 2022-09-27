// RUN: qss-opt %s -split-input-file -verify-diagnostics

// -----

pulse.sequence @sequence0 (%omega1: !quir.angle<32>, %omega2: !quir.angle<32>) {
	// expected-error@+1 {{'quir.angle_add' op is not valid within a real-time pulse sequence.}}
	%omega3 = quir.angle_add %omega1, %omega2 : !quir.angle<32>
	pulse.return
}

// -----

pulse.sequence @sequence1 (%omega1: !quir.angle<32>) {
	pulse.call_sequence @x() : () -> ()
	// expected-error@+1 {{'quir.cast' op is not valid within a real-time pulse sequence.}}
	%f1 = "quir.cast"(%omega1) : (!quir.angle<32>) -> f64
	pulse.return
}

pulse.sequence @x()

// -----

pulse.sequence @sequence2 (%cond: i1, %port : !pulse.port, %waveform: !pulse.waveform) {
	// expected-error@+1 {{'scf.yield' op is not valid within a real-time pulse sequence.}}
	scf.if %cond {
        pulse.play(%port, %waveform) : (!pulse.port, !pulse.waveform)
	}
	pulse.return
}

// -----

pulse.sequence @sequence3 () {
	// expected-error@+1 {{'quir.declare_variable' op is not valid within a real-time pulse sequence.}}
	quir.declare_variable @c1 : !quir.cbit<1>
	pulse.return
}

// -----

pulse.sequence @sequence4 () -> i1 {
	%ub = constant 10 : index
	%cmpval = constant 9 : index
	// expected-error@+1 {{'arith.cmpi' op is not valid within a real-time pulse sequence.}}
	%cond = arith.cmpi "eq", %ub, %cmpval : index
	pulse.return
}

// -----

pulse.sequence @sequence5 (%omega1: !quir.angle<32>) {
    // expected-error@+1 {{'quir.call_gate' op is not valid within a real-time pulse sequence.}}
	quir.call_gate @x() : () -> ()
	%f1 = "quir.cast"(%omega1) : (!quir.angle<32>) -> f64
	pulse.return
}

quir.circuit @x()

// -----

pulse.sequence @sequence6 () {
    // expected-error@+1 {{'pulse.create_port' op is not valid within a real-time pulse sequence.}}
	%d0 = "pulse.create_port"() {uid = "d0"} : () -> !pulse.port
	pulse.return
}

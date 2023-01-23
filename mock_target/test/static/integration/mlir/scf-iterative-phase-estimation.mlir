// RUN: qss-compiler -X=mlir --target mock --config %TEST_CFG --mock-conversion %s | FileCheck %s

// OpenQASM 3.0 Iterative Phase Estimation from Ali/Andrew Cross
// defcal y90p %0 {
//    play drive(%0), drag(...);
// }
// func @defcalRX_q0(%qa1 : !quir.qubit<1> {quir.physicalId = 0 : i32}, %theta : !quir.angle<20>) -> () attributes {quir.orig_func_name = "defcalRX"} {
func @defcalY90P_q0(%q0 : !quir.qubit<1> {quir.physicalId = 0 : i32}) -> () attributes {quir.orig_func_name = "defcalY90P"} {
    // PULSE IR Ops
    // CHECK: return
    return
}
// defcal y90p %1 {
//    play drive(%1), drag(...);
// }
func @defcalY90P_q1(%q0 : !quir.qubit<1> {quir.physicalId = 1 : i32}) -> () attributes {quir.orig_func_name = "defcalY90P"} {
    // PULSE IR Ops
    return
}
// defcal cr90p %0, %1 {
//     play flat_top_gaussian(...), drive(%0), frame(drive(%1));
// }
func @defcalCR90P_q0_q1(%q0 : !quir.qubit<1> {quir.physicalId = 0 : i32}, %q1 : !quir.qubit<1> {quir.physicalId = 1 : i32}) -> () attributes {quir.orig_func_name = "defcalCR90P"} {
    // PULSE IR Ops
    return
}
// defcal phase(angle[20]: theta) %q {
//    shift_phase drive(%q), -theta;
// }
func @defcalPhase_qq(%angle : !quir.angle, %qq : !quir.qubit<1>) -> () attributes {quir.orig_func_name = "defcalPhase"} {
    // PULSE IR Ops
    return
}
// defcal cr90m %0, $1 {
//     phase(-pi) %1;
//     cr90p %0, %1;
//     phase(pi) %1;
// }
func @defcalCR90M_q0_q1(%q0 : !quir.qubit<1> {quir.physicalId = 0 : i32}, %q1 : !quir.qubit<1> {quir.physicalId = 1 : i32}) -> () attributes {quir.orig_func_name = "defcalCR90M"} {
    %npi = quir.constant #quir.angle<-1.0  : !quir.angle<20>>
    %pi = quir.constant #quir.angle<1.0  : !quir.angle<20>>
    "quir.call_gate"(%npi, %q1) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    "quir.call_gate"(%q0, %q1) {callee = @defcalCR90P} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    "quir.call_gate"(%pi, %q1) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    return
}
// defcal x90p %q {
//     phase(pi) %q;
//     y90p %q;
//     phase(-pi) %q;
// }
func @gateX90P_qq(%qq_1 : !quir.qubit<1>) -> () attributes {quir.orig_func_name = "gateX90P"} {
    %npi = quir.constant #quir.angle<-1.0  : !quir.angle<20>>
    %pi = quir.constant #quir.angle<1.0  : !quir.angle<20>>
    "quir.call_gate"(%pi, %qq_1)  {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    "quir.call_gate"(%qq_1)       {callee = @defcalY90P}  : (!quir.qubit<1>) -> ()
    "quir.call_gate"(%npi, %qq_1) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    return
}
// defcal xp %q {
//     x90p %q;
//     x90p %q;
// }
func @gateXP_qq(%qq_1 : !quir.qubit<1>) -> () attributes {quir.orig_func_name = "gateXP"} {
    "quir.call_gate"(%qq_1) {callee = @gateX90P} : (!quir.qubit<1>) -> ()
    "quir.call_gate"(%qq_1) {callee = @gateX90P} : (!quir.qubit<1>) -> ()
    return
}
// defcal h %q {
//     phase(pi) %q;
//     y90p %q;
// }
func @gateH_qq(%qq_1 : !quir.qubit<1>) -> () attributes {quir.orig_func_name = "gateH"} {
    %pi = quir.constant #quir.angle<1.0  : !quir.angle<20>>
    "quir.call_gate"(%pi, %qq_1) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    "quir.call_gate"(%qq_1) {callee = @defcalY90P} : (!quir.qubit<1>) -> ()
    return
}
// defcal CX %control %target {
//     phase(-pi/2) %control;
//     xp %control;
//     x90p %target;
//     barrier %control, %target;
//     cr90p %control, %target;
//     barrier %control, %target;
//     xp %control;
//     barrier %control, %target;
//     cr90m %control, %target;
// }
func @gateCX_qq_qq(%ctrl : !quir.qubit<1>, %targ : !quir.qubit<1>) -> () attributes {quir.orig_func_name = "gateCX"} {
    %npi2 = quir.constant #quir.angle<-0.5  : !quir.angle<20>>
    "quir.call_gate"(%ctrl) {callee = @gateXP} : (!quir.qubit<1>) -> ()
    "quir.call_gate"(%targ) {callee = @gateX90P} : (!quir.qubit<1>) -> ()
    "quir.call_gate"(%ctrl, %targ) {callee = @defcalCR90P} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    "quir.call_gate"(%ctrl) {callee = @gateXP} : (!quir.qubit<1>) -> ()
    "quir.call_gate"(%ctrl, %targ) {callee = @defcalCR90M} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    return
}
// defcal measure %0 -> bit {
//     complex[int[24]] iq;
//     bit state;
//     complex[int[12]] k0[1024] = [i0 + q0*j, i1 + q1*j, i2 + q2*j, ...];
//     play measure(%0), flat_top_gaussian(...);
//     iq = capture acquire(%0), 2048, kernel(k0);
//     return threshold(iq, 1234);
// }
func @defcalMeasure_q0(%q0_1 : !quir.qubit<1> {quir.physicalId = 0 : i32}) -> i1 attributes {quir.orig_func_name = "defcalMeasure"} {
    // Pulse IR Ops
    %res = arith.constant false
    return %res : i1
}

// angle[3] c = 0;
oq3.declare_variable @c : !quir.cbit<3>

func @main() -> i32 {
// qubit q;
// qubit r;
%q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
%q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>


// reset q;
// reset r;
quir.reset %q0 : !quir.qubit<1>
quir.reset %q1 : !quir.qubit<1>

// h r;
quir.call_gate @gateH(%q1) : (!quir.qubit<1>) -> ()

// uint[3] power = 1;
%power1 = arith.constant 1 : i3

// for i in [0: 2] {
%lb = arith.constant 0 : index
%ub = arith.constant 3 : index
%step = arith.constant 1 : index
%power_final = scf.for %iv = %lb to %ub step %step
  iter_args(%pow_iter = %power1) -> (i3) {
  // reset q;
  quir.reset %q0 : !quir.qubit<1>
  // h q;
  quir.call_gate @gateH (%q0) : (!quir.qubit<1>) -> ()
  // cphase(power*3*pi/8) q, r;
  %pow_extended = "quir.cast"(%pow_iter) : (i3) -> !quir.angle<20>
  %ang_tpo8 = quir.constant #quir.angle<0.1875 : !quir.angle<20>>
  %phase_ang = quir.angle_mul %pow_extended, %ang_tpo8 : !quir.angle<20>
  quir.call_gate @cphase(%q0, %q1, %phase_ang) : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<20>) -> ()
  // phase(-c) q;
  %neg_ang = quir.constant #quir.angle<-1.0 : !quir.angle<20>>
  %ang_c = oq3.use_variable @c : !quir.cbit<3>
  %ang_iter_extended = "quir.cast"(%ang_c) : (!quir.cbit<3>) -> !quir.angle<20>
  %neg_ang_mul = quir.angle_mul %neg_ang, %ang_iter_extended : !quir.angle<20>
  quir.call_gate @defcalPhase(%neg_ang_mul, %q0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
  // h q;
  quir.call_gate @gateH (%q0) : (!quir.qubit<1>) -> ()
  // measure q -> c[0];
  %zeroind = arith.constant 0 : index
  %bitM_1 = quir.call_defcal_measure @defcalMeasure(%q0) : (!quir.qubit<1>) -> (i1)
  quir.assign_cbit_bit @c<3> [0] : i1 = %bitM_1
  // c <<= 1;
  %c1_i32 = arith.constant 1 : i32
  %ang_prev = oq3.use_variable @c : !quir.cbit<3>
  %ang_shifted = quir.cbit_lshift %ang_prev, %c1_i32 : (!quir.cbit<3>, i32) -> !quir.cbit<3>
  // power <<= 1;
  %c1_i3 = arith.constant 1 : i3
  %pow_shifted = arith.shli %pow_iter, %c1_i3 : i3
  scf.yield %pow_shifted : i3
// }
}
%ret = arith.constant 0 : i32
return %ret : i32
}

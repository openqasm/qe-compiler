// RUN: qss-compiler -X=mlir %s | FileCheck %s

module {
    // OpenQASM 3.0 Iterative Phase Estimation from Ali
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
    %npi2 = quir.constant #quir.angle<0.0  : !quir.angle<3>>
    // reset %0;
    // reset %1;
    %q0_0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %q1_0 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    quir.reset %q0_0 : !quir.qubit<1>
    quir.reset %q1_0 : !quir.qubit<1>
    // h %1;
    // h %0;
    "quir.call_gate"(%q1_0) {callee = @gateH} : (!quir.qubit<1>) -> ()
    "quir.call_gate"(%q0_0) {callee = @gateH} : (!quir.qubit<1>) -> ()
    // duration a, b; // should be resolved to duration by this point
    %duration_a = quir.declare_duration {value = "10ns"} : !quir.duration
    %duration_b = quir.declare_duration {value = "20ns"} : !quir.duration
    // delay(a) %0;
    // delay(b) %1;
    "quir.delay"(%duration_a, %q0_0) : (!quir.duration, !quir.qubit<1>) -> ()
    "quir.delay"(%duration_b, %q1_0) : (!quir.duration, !quir.qubit<1>) -> ()
    // cx %0, %1;
    "quir.call_gate"(%q0_0, %q1_0) {callee = @gateCX} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    // phase(1.8125*pi) %1;
    %ang1 = quir.constant #quir.angle<1.1825  : !quir.angle<20>>
    "quir.call_gate"(%ang1, %q1_0) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    // cx %0, %1;
    "quir.call_gate"(%q0_0, %q1_0) {callee = @gateCX} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    // phase(0.1875*pi) %1;
    // phase(0.1875*pi) %0;
    %ang2 = quir.constant #quir.angle<0.1875  : !quir.angle<20>>
    "quir.call_gate"(%ang1, %q1_0) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    "quir.call_gate"(%ang1, %q0_0) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    // h %0;
    "quir.call_gate"(%q0_0) {callee = @gateH} : (!quir.qubit<1>) -> ()
    // measure %0 -> c[0]; // this should not be allowed
    %zeroind = arith.constant 0 : index
    oq3.declare_variable @cbitarray : !quir.cbit<3>
    %bitM_1 = quir.call_defcal_measure @defcalMeasure(%q0_0) : (!quir.qubit<1>) -> (i1)
    quir.assign_cbit_bit @cbitarray<3> [0] : i1 = %bitM_1
    // c <<= 1;
    %c1_i32 = arith.constant 1 : i32
    %creg_X = quir.use_variable @cbitarray : !quir.cbit<3>
    %creg_1 = quir.cbit_lshift %creg_X, %c1_i32 : (!quir.cbit<3>, i32) -> !quir.cbit<3>
    // reset %0;
    quir.reset %q0_0 : !quir.qubit<1>
    // h %0;
    "quir.call_gate"(%q0_0) {callee = @gateH} : (!quir.qubit<1>) -> ()
    // duration cs;
    // delay(cs) %1;
    %duration_c = quir.declare_duration {value = "5ns"} : !quir.duration
    "quir.delay"(%duration_c, %q1_0) : (!quir.duration, !quir.qubit<1>) -> ()
    // cx %0, %1;
    "quir.call_gate"(%q0_0, %q1_0) {callee = @gateCX} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    // phase(1.625*pi) %1;  // mod 2*pi
    %ang3 = quir.constant #quir.angle<1.625  : !quir.angle<20>>
    "quir.call_gate"(%ang3, %q1_0) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    // cx %0, %1;
    "quir.call_gate"(%q0_0, %q1_0) {callee = @gateCX} : (!quir.qubit<1>, !quir.qubit<1>) -> ()
    // phase(0.375*pi) %1;
    %ang4 = quir.constant #quir.angle<0.375  : !quir.angle<20>>
    "quir.call_gate"(%ang4, %q1_0) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    // angle[32] temp_1 = 0.375*pi;
    // temp_1 -= c;  // cast and do arithmetic mod 2 pi
    %tmp_angle_1 = quir.constant #quir.angle<0.375  : !quir.angle<32>>
    %cast_c = "quir.cast"(%creg_1) : (!quir.cbit<3>) -> !quir.angle<32>
    // Math Ops resulting in !quir.angle<32>
    %tmp_angle_2 = quir.constant #quir.angle<0.0  : !quir.angle<32>>
    %tmp_angle_2_cast = "quir.cast"(%tmp_angle_2) : (!quir.angle<32>) -> !quir.angle<20>
    // phase(temp_1) %0;
    "quir.call_gate"(%tmp_angle_2_cast, %q1_0) {callee = @defcalPhase} : (!quir.angle<20>, !quir.qubit<1>) -> ()
    // h %0;
    "quir.call_gate"(%q0_0) {callee = @gateH} : (!quir.qubit<1>) -> ()
    // measure %0 -> c[0];
    %bitM_2 = quir.call_defcal_measure @defcalMeasure(%q0_0) : (!quir.qubit<1>) -> (i1)
    // recent MLIR releases can express "tensor insert element", which we code around here:
    oq3.assign_variable @cbitarray : !quir.cbit<3> = %creg_1
    quir.assign_cbit_bit @cbitarray<3> [0] : i1 = %bitM_2
    %creg_2 = quir.use_variable @cbitarray : !quir.cbit<3>
    // c <<= 1;
    %creg_3 = quir.cbit_lshift %creg_2, %c1_i32 : (!quir.cbit<3>, i32) -> !quir.cbit<3>
    // reset %0;
    // h %0;
    // duration d;
    // delay(d) %1;
    // cx %0, %1;
    // phase(1.25*pi) %1;  // mod 2*pi
    // cx %0, %1;
    // phase(0.75*pi) %1;
    // angle[32] temp_2 = 0.75*pi;
    // temp_2 -= c;  // cast and do arithmetic mod 2 pi
    // phase(temp_2) %0;
    // h %0;
    // measure %0 -> c[0];
    // c <<= 1;
    // duration f;
    // delay(f) %1;
}

// RUN: qss-compiler -X=mlir --quir-arg-specialization %s | FileCheck %s

// CHECK: func @"defcalPhase_q0_!quir.angle<20>_!quir.qubit<1>"(%arg0: !quir.angle<20>, %arg1: !quir.qubit<1> {quir.physicalId = 0 : i32}) attributes {quir.orig_func_name = "defcalPhase"}
// CHECK: func @"defcalPhase_q0_!quir.angle<10>_!quir.qubit<1>"(%arg0: !quir.angle<10>, %arg1: !quir.qubit<1> {quir.physicalId = 0 : i32}) attributes {quir.orig_func_name = "defcalPhase"}
func @defcalPhase_q0(%arg0: !quir.angle, %arg1: !quir.qubit<1> {quir.physicalId = 0 : i32}) attributes {quir.orig_func_name = "defcalPhase"} {
  return
}

// CHECK: func @"subroutine1_!quir.qubit<1>_!quir.angle<20>_index"(%arg0: !quir.qubit<1>, %arg1: !quir.angle<20>, %arg2: index)
func @subroutine1(%q1 : !quir.qubit<1>, %phi : !quir.angle, %ub : index) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    quir.call_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
    quir.call_defcal_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
    %res = quir.call_defcal_measure @defcalMeasure_q0(%q1, %phi) : (!quir.qubit<1>, !quir.angle) -> i1
    scf.if %res {
      quir.call_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
    }
  }
  return
}

// CHECK: func @"defcalMeasure_q0_!quir.qubit<1>_!quir.angle<10>"(%arg0: !quir.qubit<1> {quir.physicalId = 0 : i32}, %arg1: !quir.angle<10>) -> i1 attributes {quir.orig_func_name = "defcalMeasure"}
// CHECK: func @"defcalMeasure_q0_!quir.qubit<1>_!quir.angle<20>"(%arg0: !quir.qubit<1> {quir.physicalId = 0 : i32}, %arg1: !quir.angle<20>) -> i1 attributes {quir.orig_func_name = "defcalMeasure"}
func @defcalMeasure_q0(%q1: !quir.qubit<1> {quir.physicalId = 0 : i32}, %phi : !quir.angle) -> i1 attributes {quir.orig_func_name = "defcalMeasure"} {
  quir.call_gate @defcalPhase_q0(%phi, %q1) : (!quir.angle, !quir.qubit<1>) -> ()
  %false = arith.constant false
  return %false : i1
}

func @main () -> i32 {
  %q1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %ang = quir.constant #quir.angle<0.1 : !quir.angle<20>>
  %ub = arith.constant 10 : index
  // CHECK: quir.call_subroutine @"subroutine1_!quir.qubit<1>_!quir.angle<20>_index"(%{{.*}}, %{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
  quir.call_subroutine @subroutine1(%q1, %ang, %ub) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
  // CHECK: quir.call_defcal_gate @"defcalPhase_q0_!quir.angle<20>_!quir.qubit<1>"(%{{.*}}, %{{.*}}) : (!quir.angle<20>, !quir.qubit<1>) -> ()
  quir.call_defcal_gate @defcalPhase_q0(%ang, %q1) : (!quir.angle<20>, !quir.qubit<1>) -> ()
  %cmpval = arith.constant 9 : index
  %cond = arith.cmpi "eq", %ub, %cmpval : index
  scf.if %cond {
    // CHECK: quir.call_gate @"defcalPhase_q0_!quir.angle<20>_!quir.qubit<1>"(%{{.*}}, %{{.*}}) : (!quir.angle<20>, !quir.qubit<1>) -> ()
    quir.call_gate @defcalPhase_q0(%ang, %q1) : (!quir.angle<20>, !quir.qubit<1>) -> ()
    quir.call_subroutine @subroutine1(%q1, %ang, %ub) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
    %true = arith.constant true
    scf.if %true {
      %ang2 = quir.constant #quir.angle<0.2 : !quir.angle<10>>
      // CHECK: quir.call_defcal_gate @"defcalPhase_q0_!quir.angle<10>_!quir.qubit<1>"(%{{.*}}, %{{.*}}) : (!quir.angle<10>, !quir.qubit<1>) -> ()
      quir.call_defcal_gate @defcalPhase_q0(%ang2, %q1) : (!quir.angle<10>, !quir.qubit<1>) -> ()
      // CHECK: %{{.*}} = quir.call_defcal_measure @"defcalMeasure_q0_!quir.qubit<1>_!quir.angle<10>"(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.angle<10>) -> i1
      %res = quir.call_defcal_measure @defcalMeasure_q0(%q1, %ang2) : (!quir.qubit<1>, !quir.angle<10>) -> i1
    }
  }
  %zero = arith.constant 0 : i32
  return %zero : i32
}

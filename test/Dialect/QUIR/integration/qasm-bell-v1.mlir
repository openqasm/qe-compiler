// RUN: qss-compiler -X=mlir %s | FileCheck %s

// gate h q {
//     U(1.57079632679, 0.0, 3.14159265359) q;
// }
func @h (%q : !quir.qubit<1>) -> () {
    %a0 = quir.constant #quir.angle<1.57079632679 : !quir.angle<20>>
    %a1 = quir.constant #quir.angle<0.0 : !quir.angle<20>>
    %a2 = quir.constant #quir.angle<3.14159265359 : !quir.angle<20>>
    quir.builtin_U %q, %a0, %a1, %a2 : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
    return
}

func @main () -> i32 {
    // qubit %0;
    // qubit %1;
    // CHECK: %{{.*}} = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
    %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
    // h %0;
    quir.call_gate @h(%q0) : (!quir.qubit<1>) -> ()
    // CX %0, %1;
    quir.builtin_CX %q0, %q1 : !quir.qubit<1>, !quir.qubit<1>
    // measure %0 -> c0 and measure %1 -> c1 only in qasm-bell-v0.mlir
    %zero = arith.constant 0 : i32
    return %zero : i32
}

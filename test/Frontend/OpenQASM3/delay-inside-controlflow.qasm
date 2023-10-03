OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

gate x q {}
qubit $0;
bit c;

// MLIR-CIRCUITS:   quir.circuit @circuit_0(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.duration) {
// MLIR-CIRCUITS:     quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     %cst = constant unit
// MLIR-CIRCUITS:     quir.return
// MLIR-CIRCUITS:   }
// MLIR-CIRCUITS:   quir.circuit @circuit_1(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.duration) {
// MLIR-CIRCUITS:     quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     %cst = constant unit
// MLIR-CIRCUITS:     quir.return
// MLIR-CIRCUITS:   }
// MLIR-CIRCUITS:   quir.circuit @circuit_2(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.duration) {
// MLIR-CIRCUITS:     quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     %cst = constant unit
// MLIR-CIRCUITS:     quir.return
// MLIR-CIRCUITS:   }
// MLIR-CIRCUITS:   quir.circuit @circuit_3(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.duration) {
// MLIR-CIRCUITS:     quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     %cst = constant unit
// MLIR-CIRCUITS:     quir.return
// MLIR-CIRCUITS:   }
// MLIR-CIRCUITS:   quir.circuit @circuit_4(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.duration) {
// MLIR-CIRCUITS:     quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     %cst = constant unit
// MLIR-CIRCUITS:     quir.return
// MLIR-CIRCUITS:   }
// MLIR-CIRCUITS:   quir.circuit @circuit_5(%{{.*}}: !quir.qubit<1>, %{{.*}}: !quir.duration) {
// MLIR-CIRCUITS:     quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     quir.call_gate @x(%{{.*}}) : (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:     %cst = constant unit
// MLIR-CIRCUITS:     quir.return
// MLIR-CIRCUITS:   }

c = 0;
// MLIR: scf.if %{{.*}} {
if (c == 0) {
// MLIR:             %{{.*}} = quir.constant #quir.duration<16 : !quir.duration<dt>>
// MLIR-NO-CIRCUITS: quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:    quir.call_circuit @circuit_0(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.duration) -> ()
  delay[16dt] $0;
  x $0;
// MLIR: } else {
} else {
// MLIR:             %{{.*}} = quir.constant #quir.duration<96 : !quir.duration<dt>>
// MLIR-NO-CIRCUITS: quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:    quir.call_circuit @circuit_1(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.duration) -> ()
  delay[96dt] $0;
  x $0;
}

// MLIR: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %c{{.*}}1_3 {
for ii in [0 : 4] {
// MLIR:             %{{.*}} = quir.constant #quir.duration<32 : !quir.duration<dt>>
// MLIR-NO-CIRCUITS: quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:    quir.call_circuit @circuit_2(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.duration) -> ()
  delay[32dt] $0;
  x $0;
}

// MLIR: scf.while : () -> () {
int nn = 1;
while (nn != 0) {
// MLIR:             %{{.*}} = quir.constant #quir.duration<48 : !quir.duration<dt>>
// MLIR-NO-CIRCUITS: quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:    quir.call_circuit @circuit_3(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.duration) -> ()
  delay[48dt] $0;
  x $0;
  nn = 0;
}

int ii = 15;
// MLIR: quir.switch %{{.*}}{
switch (ii) {

// default case gets printed first
// MLIR:             %{{.*}} = quir.constant #quir.duration<80 : !quir.duration<dt>>
// MLIR-NO-CIRCUITS: quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:    quir.call_circuit @circuit_4(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.duration) -> ()


// MLIR:       }[1 : {
      case 1: {
      delay[64dt] $0;
// MLIR:             %{{.*}} = quir.constant #quir.duration<64 : !quir.duration<dt>>
// MLIR-NO-CIRCUITS: quir.delay %{{.*}}, (%{{.*}}) : !quir.duration, (!quir.qubit<1>) -> ()
// MLIR-CIRCUITS:    quir.call_circuit @circuit_5(%{{.*}}, %{{.*}}) : (!quir.qubit<1>, !quir.duration) -> ()
      x $0;
    }
    break;
    default: {
      delay[80dt] $0;
      x $0;
    }
    break;
}

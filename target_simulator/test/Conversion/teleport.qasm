// RUN: touch test.cfg
// RUN: qss-compiler -X=qasm --emit=mlir --target simulator --config test.cfg %s --num-shots=1 --break-reset --simulator-output-cregs --quir-eliminate-variables --simulator-quir-to-aer | FileCheck %s

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//

OPENQASM 3.0;

gate cx control, target { }

qubit $0;
qubit $1;
qubit $2;
bit[3] c;
// Allocations of classical bits are moved forward
// CHECK: {{.*}} = memref.alloca() : memref<i3>
// CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
// CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
// CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}

U(0.3, 0.2, 0.1) $0;
// CHECK: llvm.call @aer_apply_u3{{.*$}}

U(1.57079632679, 0, 3.14159265359) $1; // h $1;
cx $1, $2;
cx $0, $1;
U(1.57079632679, 0, 3.14159265359) $0; // h $0;
// CHECK: llvm.call @aer_apply_u3{{.*$}}
// CHECK: llvm.call @aer_apply_cx{{.*$}}
// CHECK: llvm.call @aer_apply_cx{{.*$}}
// CHECK: llvm.call @aer_apply_u3{{.*$}}

c[0] = measure $0;
c[1] = measure $1;
// CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
// CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
// Consecutive store to classical registers are merged:
// CHECK: affine.store %{{[0-9]+}}, %{{[0-9]+}}[] : memref<i3>

if (c[0]==1) {
// CHECK: scf.if %{{[0-9]+}} {
  U(0, 0, 3.14159265359) $2; // z $2;
  // CHECK: llvm.call @aer_apply_u3{{.*$}}
}

if (c[1]==1) {
// CHECK: scf.if %{{[0-9]+}} {
  U(3.14159265359, 0, 3.14159265359) $2; // x $2;
  // CHECK: llvm.call @aer_apply_u3{{.*$}}
}

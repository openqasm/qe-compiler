// RUN: touch test.cfg
// RUN: echo "{}" >> test.cfg
// RUN: qss-compiler -X=qasm --emit=mlir --target aer-simulator --config test.cfg %s --num-shots=1 --simulator-output-cregs --quir-eliminate-variables --simulator-quir-to-aer | FileCheck %s

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

gate cx control, target {}

qubit $0;
qubit $1;
bit c0;
bit c1;
// Allocations of classical bits are moved forward
// CHECK: {{.*}} = memref.alloca() : memref<i1>
// CHECK: {{.*}} = memref.alloca() : memref<i1>
// CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}
// CHECK: {{.*}} = llvm.call @aer_allocate_qubits{{.*$}}



U(1.57079632679, 0.0, 3.14159265359) $0;
cx $0, $1;
// CHECK: llvm.call @aer_apply_u3{{.*$}}
// CHECK: llvm.call @aer_apply_cx{{.*$}}

measure $0 -> c0;
measure $1 -> c1;
// CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
// CHECK: affine.store %{{[0-9]+}}, %{{[0-9]+}}[] : memref<i1>
// CHECK: {{.*}} = llvm.call @aer_apply_measure{{.*$}}
// CHECK: affine.store %{{[0-9]+}}, %{{[0-9]+}}[] : memref<i1>

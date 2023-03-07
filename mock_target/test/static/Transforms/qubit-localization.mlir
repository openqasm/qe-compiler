// RUN: qss-compiler -X=mlir --target mock --config %TEST_CFG --mock-conversion %s | FileCheck %s

// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

func private @kernel1 (%c0 : i1, %c1 : i1, %c2 : i1) -> i1

func @gateH_qq(%qArg : !quir.qubit<1>) attributes {quir.orig_func_name = "gateH"} {
  %ang = quir.constant #quir.angle<0.1 : !quir.angle<20>>
  quir.builtin_U %qArg, %ang, %ang, %ang : !quir.qubit<1>, !quir.angle<20>, !quir.angle<20>, !quir.angle<20>
  return
}

func @subroutine1(%qq1 : !quir.qubit<1>, %phi : !quir.angle, %ub : index) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    quir.call_gate @defcalPhase_q0(%phi, %qq1) : (!quir.angle, !quir.qubit<1>) -> ()
    quir.call_defcal_gate @defcalPhase_q0(%phi, %qq1) : (!quir.angle, !quir.qubit<1>) -> ()
    %res = quir.call_defcal_measure @defcalMeasure_q0(%qq1, %phi) : (!quir.qubit<1>, !quir.angle) -> i1
    scf.if %res {
      quir.call_gate @defcalPhase_q0(%phi, %qq1) : (!quir.angle, !quir.qubit<1>) -> ()
    }
  }
  return
}

func @subroutine2(%qq1 : !quir.qubit<1>, %qq2 : !quir.qubit<1>) {
  %zero = quir.constant #quir.angle<0.0 : !quir.angle<20>>
  quir.call_gate @defcalPhase_qq(%zero, %qq1) : (!quir.angle<20>, !quir.qubit<1>) -> ()
  quir.call_gate @defcalPhase_qq(%zero, %qq2) : (!quir.angle<20>, !quir.qubit<1>) -> ()
  %ub = arith.constant 5 : index
  quir.call_subroutine @subroutine1(%qq1, %zero, %ub) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
  return
}

func @main () -> i32 {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>

  quir.reset %q0 : !quir.qubit<1>
  quir.barrier %q0, %q1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
  quir.call_gate @gateH(%q0) : (!quir.qubit<1>) -> ()
  %ang1 = quir.constant #quir.angle<0.5 : !quir.angle<20>>
  %ub = arith.constant 10 : index
  quir.call_subroutine @subroutine1(%q1, %ang1, %ub) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
  quir.call_subroutine @subroutine1(%q0, %ang1, %ub) : (!quir.qubit<1>, !quir.angle<20>, index) -> ()
  quir.call_subroutine @subroutine2(%q0, %q1) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
  quir.call_subroutine @subroutine2(%q1, %q0) : (!quir.qubit<1>, !quir.qubit<1>) -> ()
  %c0 = quir.call_defcal_measure @defcalMeasure_q0(%q0) : (!quir.qubit<1>) -> i1
  %c1 = quir.call_defcal_measure @defcalMeasure_q1(%q0) : (!quir.qubit<1>) -> i1
  %res = quir.call_kernel @kernel1(%c0, %c1, %c0) : (i1, i1, i1) -> i1
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// CHECK:   module @controller attributes {quir.nodeId = 1000 : i32, quir.nodeType = "controller"} {
// CHECK:     func private @kernel1(i1, i1, i1) -> i1 attributes {quir.classicalOnly = true}
// CHECK:     func @subroutine2_q1_q0() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @subroutine2_q0_q1() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @"subroutine1_q0_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @subroutine1_q0(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @"subroutine1_q1_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @subroutine1_q1(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @main() -> i32 attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @subroutine2_q0_q1() : () -> ()
// CHECK:       quir.call_subroutine @subroutine2_q1_q0() : () -> ()
// CHECK:       %2 = quir.call_kernel @kernel1(%0, %1, %0) : (i1, i1, i1) -> i1
// CHECK:   module @mock_drive_0 attributes {quir.nodeId = 1 : i32, quir.nodeType = "drive", quir.physicalId = 0 : i32} {
// CHECK:     func @gateH_q0(%arg0: !quir.qubit<1> {quir.physicalId = 0 : i32}) attributes {quir.classicalOnly = false, quir.orig_func_name = "gateH"} {
// CHECK:     func @subroutine2_q1_q0() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_gate @defcalPhase_qq(%angle, %1) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @subroutine2_q0_q1() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_gate @defcalPhase_qq(%angle, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @"subroutine1_q0_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:         quir.call_defcal_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle<20>) -> i1
// CHECK:           quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:     func @subroutine1_q0(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle, !quir.qubit<1>) -> ()
// CHECK:         quir.call_defcal_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle, !quir.qubit<1>) -> ()
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle) -> i1
// CHECK:           quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle, !quir.qubit<1>) -> ()
// CHECK:     func @"subroutine1_q1_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @subroutine1_q1(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @main() -> i32 attributes {quir.classicalOnly = false} {
// CHECK:       quir.barrier %0, %1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
// CHECK:       quir.call_gate @gateH_q0(%0) : (!quir.qubit<1>) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @subroutine2_q0_q1() : () -> ()
// CHECK:       quir.call_subroutine @subroutine2_q1_q0() : () -> ()
// CHECK:       %2 = quir.call_defcal_measure @defcalMeasure_q0(%0) : (!quir.qubit<1>) -> i1
// CHECK:       %3 = quir.call_defcal_measure @defcalMeasure_q1(%0) : (!quir.qubit<1>) -> i1
// CHECK:   module @mock_acquire_0 attributes {quir.nodeId = 0 : i32, quir.nodeType = "acquire", quir.physicalIds = [0 : i32, 1 : i32]} {
// CHECK:     func @subroutine2_q1_q0() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @subroutine2_q0_q1() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @"subroutine1_q0_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle<20>) -> i1
// CHECK:     func @subroutine1_q0(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle) -> i1
// CHECK:     func @"subroutine1_q1_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle<20>) -> i1
// CHECK:     func @subroutine1_q1(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle) -> i1
// CHECK:     func @main() -> i32 attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @subroutine2_q0_q1() : () -> ()
// CHECK:       quir.call_subroutine @subroutine2_q1_q0() : () -> ()
// CHECK:       %2 = quir.call_defcal_measure @defcalMeasure_q0(%0) : (!quir.qubit<1>) -> i1
// CHECK:       %3 = quir.call_defcal_measure @defcalMeasure_q1(%0) : (!quir.qubit<1>) -> i1
// CHECK:   module @mock_drive_1 attributes {quir.nodeId = 2 : i32, quir.nodeType = "drive", quir.physicalId = 1 : i32} {
// CHECK:     func @subroutine2_q1_q0() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_gate @defcalPhase_qq(%angle, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @subroutine2_q0_q1() attributes {quir.classicalOnly = false} {
// CHECK:       quir.call_gate @defcalPhase_qq(%angle, %1) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c5) : (!quir.angle<20>, index) -> ()
// CHECK:     func @"subroutine1_q0_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @subroutine1_q0(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:     func @"subroutine1_q1_!quir.angle<20>_index"(%arg0: !quir.angle<20>, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:         quir.call_defcal_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle<20>) -> i1
// CHECK:           quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle<20>, !quir.qubit<1>) -> ()
// CHECK:     func @subroutine1_q1(%arg0: !quir.angle, %arg1: index) attributes {quir.classicalOnly = false} {
// CHECK:         quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle, !quir.qubit<1>) -> ()
// CHECK:         quir.call_defcal_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle, !quir.qubit<1>) -> ()
// CHECK:         %1 = quir.call_defcal_measure @defcalMeasure_q0(%0, %arg0) : (!quir.qubit<1>, !quir.angle) -> i1
// CHECK:           quir.call_gate @defcalPhase_q0(%arg0, %0) : (!quir.angle, !quir.qubit<1>) -> ()
// CHECK:     func @main() -> i32 attributes {quir.classicalOnly = false} {
// CHECK:       quir.barrier %0, %1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q1_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @"subroutine1_q0_!quir.angle<20>_index"(%angle, %c10) : (!quir.angle<20>, index) -> ()
// CHECK:       quir.call_subroutine @subroutine2_q0_q1() : () -> ()
// CHECK:       quir.call_subroutine @subroutine2_q1_q0() : () -> ()

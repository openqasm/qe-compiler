// RUN: qss-opt %s -split-input-file -verify-diagnostics

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

// -----

func.func @qubit_type_parse_error() {
  // expected-error@+1 {{width must be > 0}}
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<0>
  return
}

// -----

func.func @angle_type_parse_error() {
  // expected-error@+1 {{width must be > 0}}
  %a1 = quir.constant #quir.angle<1.0> : !quir.angle<0>
  return
}

// -----

func.func @angle_type_cmp_error() {
  %a1 = quir.constant #quir.angle<0.0> : !quir.angle<20>
  %a2 = quir.constant #quir.angle<0.0> : !quir.angle<20>
  // expected-error@+1 {{'oq3.angle_cmp' op requires predicate "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"}}
  %b = oq3.angle_cmp {predicate = "test"} %a1, %a2 : !quir.angle<20> -> i1
  return
}

// -----

func.func @call_defcal_measure_no_qubit_args() {
  %c1 = quir.constant #quir.angle<1.0> : !quir.angle<20>
  // expected-error@+1 {{'quir.call_defcal_measure' op requires exactly one qubit}}
  %a1 = quir.call_defcal_measure @proto1(%c1) : (!quir.angle<20>) -> i1
  return
}

// -----

func.func @call_defcal_measure_too_many_qubit_args() {
  %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %q1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
  // expected-error@+1 {{'quir.call_defcal_measure' op requires exactly one qubit argument}}
  %a1 = quir.call_defcal_measure @proto1(%q0, %q1) : (!quir.qubit<1>, !quir.qubit<1>) -> i1
  return
}

// -----

func.func @quir_switch (%flag: i32) -> (i32) {
  // expected-error@+1 {{expected '{' to begin a region}}
  %y = quir.switch %flag -> (i32) [
    4: {
      %y_1 = arith.constant 1 : i32
      quir.yield %y_1 : i32
    }
  ]
  return %y : i32
}

// -----

func.func @quir_switch (%flag: i32) -> (i32) {
  %y = quir.switch %flag -> (i32)
  {
    %y_default = arith.constant 0 : i32
    // expected-error@+1 {{'quir.yield' op parent of yield must have same number of results as the yield operands}}
    quir.yield
  } [
    1: {
      %y_1 = arith.constant 1 : i32
      quir.yield %y_1 : i32
    }
  ]
  return %y : i32
}

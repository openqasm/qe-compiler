//===- PulseTraits.td - Pulse dialect traits -*- C++ -*-======================//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  Traits for the Pulse dialect
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_TRAITS
#define PULSE_TRAITS

include "mlir/IR/OpBase.td"

// Pulse Traits

/// Operation is allowed within a real-time scheduled pulse.sequence
def SequenceAllowed : NativeOpTrait<"SequenceAllowed">{
    let cppNamespace= "::mlir::pulse";
}

/// Operation must be within a real-time scheduled pulse.sequence
def SequenceRequired : NativeOpTrait<"SequenceRequired">{
    let cppNamespace= "::mlir::pulse";
}

/// Operation receives a target argument
def HasTargetFrame : NativeOpTrait<"HasTargetFrame">{
    let cppNamespace= "::mlir::pulse";
}

#endif // PULSE_TRAITS

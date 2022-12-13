//===- PulseTraits.td - Pulse dialect traits -*- C++ -*-======================//
//
// (C) Copyright IBM 2022.
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

/// Operation is a allowed within a real-time scheduled pulse.sequence
def SequenceAllowed : NativeOpTrait<"SequenceAllowed">{
    let cppNamespace= "::mlir::pulse";
}

#endif // PULSE_TRAITS

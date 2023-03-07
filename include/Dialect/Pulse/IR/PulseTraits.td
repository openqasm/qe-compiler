//===- PulseTraits.td - Pulse dialect traits -*- C++ -*-======================//
//
// (C) Copyright IBM 2022, 2023.
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

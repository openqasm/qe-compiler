//===- PulseTraits.h - Pulse dialect traits -*- C++ -*-=======================//
//
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
//
//===----------------------------------------------------------------------===//
///
///  Traits for the Pulse dialect
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_PULSETRAITS_H
#define PULSE_PULSETRAITS_H

#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// Operation Trait Types
//===----------------------------------------------------------------------===//

namespace mlir::pulse {

template <typename ConcreteType>
class SequenceAllowed
    : public mlir::OpTrait::TraitBase<ConcreteType, SequenceAllowed> {};

template <typename ConcreteType>
class SequenceRequired
    : public mlir::OpTrait::TraitBase<ConcreteType, SequenceRequired> {
public:
  static LogicalResult verifyTrait(Operation *op);
};

template <typename ConcreteType>
class HasTargetFrame
    : public mlir::OpTrait::TraitBase<ConcreteType, HasTargetFrame> {};

} // namespace mlir::pulse

#endif // PULSE_PULSETRAITS_H

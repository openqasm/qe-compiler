//===- PulseTraits.h - Pulse dialect traits -*- C++ -*-=======================//
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

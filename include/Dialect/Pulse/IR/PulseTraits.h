//===- PulseTraits.h - Pulse dialect traits -*- C++ -*-=======================//
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

} // namespace mlir::pulse

#endif // PULSE_PULSETRAITS_H

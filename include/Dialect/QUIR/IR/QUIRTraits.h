//===- QUIRTraits.h - QUIR dialect traits -*- C++ -*-=========================//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  Traits for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUIRTRAITS_H
#define QUIR_QUIRTRAITS_H

#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// Operation Trait Types
//===----------------------------------------------------------------------===//

namespace mlir::quir {

template <typename ConcreteType>
class CPTPOp : public mlir::OpTrait::TraitBase<ConcreteType, CPTPOp> {};

template <typename ConcreteType>
class UnitaryOp : public mlir::OpTrait::TraitBase<ConcreteType, UnitaryOp> {};

} // namespace mlir::quir

#endif // QUIR_QUIRTRAITS_H

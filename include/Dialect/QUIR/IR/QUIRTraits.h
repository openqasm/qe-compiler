//===- QUIRTraits.h - QUIR dialect traits -*- C++ -*-=========================//
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

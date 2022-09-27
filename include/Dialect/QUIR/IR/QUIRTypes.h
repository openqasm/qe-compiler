//===- QUIRTypes.h - QUIR dialect types -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUIRTYPES_H
#define QUIR_QUIRTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/QUIR/IR/QUIRTypes.h.inc"

namespace mlir::quir {

/// Check whether a type is a bool (i1)
inline bool isBoolType(mlir::Type type) { return type.isInteger(1); }

} // namespace mlir::quir

#endif // QUIR_QUIRTYPES_H

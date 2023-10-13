//===- QUIRTypes.h - QUIR dialect types -------------------------*- C++ -*-===//
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

#ifndef QUIR_QUIRTYPES_H
#define QUIR_QUIRTYPES_H

#include "Dialect/QUIR/IR/QUIREnums.h"

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

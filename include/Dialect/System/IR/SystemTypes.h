//===- SystemTypes.h - System dialect types ---------------*- C++ -*-=========//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEM_SYSTEMTYPES_H
#define SYSTEM_SYSTEMTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/System/IR/SystemTypes.h.inc"

namespace mlir::sys {} // namespace mlir::sys

#endif // SYSTEM_SYSTEMTYPES_H

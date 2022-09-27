//===- PulseTypes.h - Pulse dialect ops -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef PULSE_PULSETYPES_H
#define PULSE_PULSETYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Pulse/IR/PulseTypes.h.inc"

#endif // PULSE_PULSETYPES_H

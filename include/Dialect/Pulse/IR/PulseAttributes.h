//===- PulseAttributes.h - Pulse dialect attributes -------------*- C++ -*-===//
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
///  Attributes for the Pulse dialect
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_PULSEATTRIBUTES_H
#define PULSE_PULSEATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::quir {} // namespace mlir::quir

#define GET_ATTRDEF_CLASSES
#include "Dialect/Pulse/IR/PulseAttributes.h.inc"

#endif // PULSE_PULSEATTRIBUTES_H

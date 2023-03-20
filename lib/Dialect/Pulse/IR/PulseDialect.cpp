//===- PulseDialect.cpp - Pulse dialect -------------------------*- C++ -*-===//
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
///  This file defines the Pulse dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTypes.h"

#include "mlir/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

/// Tablegen Definitions
#include "Dialect/Pulse/IR/PulseDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Pulse/IR/PulseTypes.cpp.inc"

namespace mlir::pulse {

void pulse::PulseDialect::initialize() {

  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Pulse/IR/PulseTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/Pulse/IR/Pulse.cpp.inc"
      >();
}

} // namespace mlir::pulse

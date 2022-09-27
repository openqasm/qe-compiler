//===- PulseDialect.cpp - Pulse dialect -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
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

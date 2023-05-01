//===- PulseInterfaces.cpp - Pulse dialect interfaces ---------- *- C++ -*-===//
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
///  This file defines the Pulse dialect interfaces.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseInterfaces.h"

using namespace mlir::pulse;

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// PulseOpSchedulingInterface
//===----------------------------------------------------------------------===//

int64_t interfaces_impl::getTimepoint(mlir::Operation *op) {
  if (op->hasAttr("pulse.timepoint"))
    return op->getAttrOfType<IntegerAttr>("pulse.timepoint").getInt();
  return 0;
}

void interfaces_impl::setTimepoint(mlir::Operation *op, IntegerAttr timepointAttr) {
  op->setAttr("pulse.timepoint", timepointAttr);
}

int64_t interfaces_impl::getSetupLatency(Operation *op) {
  if (op->hasAttr("pulse.setupLatency"))
    return op->getAttrOfType<IntegerAttr>("pulse.setupLatency").getInt();
  return 0;
}

void interfaces_impl::setSetupLatency(Operation *op, IntegerAttr setupLatencyAttr) {
  op->setAttr("pulse.setupLatency", setupLatencyAttr);
}
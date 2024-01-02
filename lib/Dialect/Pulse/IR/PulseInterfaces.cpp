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

#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir::pulse;

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// PulseOpSchedulingInterface
//===----------------------------------------------------------------------===//

llvm::Optional<int64_t> interfaces_impl::getTimepoint(mlir::Operation *op) {
  if (op->hasAttr("pulse.timepoint"))
    return op->getAttrOfType<IntegerAttr>("pulse.timepoint").getInt();
  return llvm::None;
}

void interfaces_impl::setTimepoint(mlir::Operation *op, int64_t timepoint) {
  mlir::OpBuilder builder(op);
  op->setAttr("pulse.timepoint", builder.getI64IntegerAttr(timepoint));
}

// MLIR does does not have a setUI64IntegerAttr so duration and setup latency
// are stored as I64IntegerAttr but should be treated as a uint64_t
llvm::Optional<uint64_t> interfaces_impl::getSetupLatency(Operation *op) {
  if (op->hasAttr("pulse.setupLatency"))
    return static_cast<uint64_t>(
        op->getAttrOfType<IntegerAttr>("pulse.setupLatency").getInt());
  return llvm::None;
}

void interfaces_impl::setSetupLatency(Operation *op, uint64_t setupLatency) {
  mlir::OpBuilder builder(op);
  op->setAttr("pulse.setupLatency", builder.getI64IntegerAttr(setupLatency));
}

llvm::Expected<uint64_t>
interfaces_impl::getDuration(Operation *op, Operation *callSequenceOp) {
  if (op->hasAttr("pulse.duration"))
    return static_cast<uint64_t>(
        op->getAttrOfType<IntegerAttr>("pulse.duration").getInt());
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Operation does not have a pulse.duration attribute.");
}

llvm::Expected<mlir::ArrayAttr> interfaces_impl::getPorts(mlir::Operation *op) {
  if (op->hasAttrOfType<ArrayAttr>("pulse.argPorts"))
    return op->getAttrOfType<ArrayAttr>("pulse.argPorts");
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Operation does not have a pulse.argPorts attribute.");
}

void interfaces_impl::setDuration(Operation *op, uint64_t duration) {
  mlir::OpBuilder builder(op);
  op->setAttr("pulse.duration", builder.getI64IntegerAttr(duration));
}

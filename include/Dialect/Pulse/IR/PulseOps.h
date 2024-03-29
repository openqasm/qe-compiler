//===- PulseOps.h - Pulse dialect ops ---------------------------*- C++ -*-===//
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

#ifndef PULSE_PULSEOPS_H
#define PULSE_PULSEOPS_H

#include "Dialect/Pulse/IR/PulseAttributes.h"
#include "Dialect/Pulse/IR/PulseInterfaces.h"
#include "Dialect/Pulse/IR/PulseTraits.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/Support/Error.h"

#define GET_OP_CLASSES
#include "Dialect/Pulse/IR/Pulse.h.inc"

namespace mlir::pulse {

// define SequenceRequired::verifyTrait here rather than in PulseTraits.h
// in order to prevent circular header dependencies
template <typename ConcreteType>
LogicalResult SequenceRequired<ConcreteType>::verifyTrait(Operation *op) {
  if (isa<SequenceOp>(op->getParentOp()))
    return success();
  return op->emitOpError() << "expects parent op '"
                           << SequenceOp::getOperationName() << "'";
}

} // namespace mlir::pulse

#endif // PULSE_PULSEOPS_H

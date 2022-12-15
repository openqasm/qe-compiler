//===- PulseOps.h - Pulse dialect ops ---------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef PULSE_PULSEOPS_H
#define PULSE_PULSEOPS_H

#include "Dialect/Pulse/IR/PulseTraits.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
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

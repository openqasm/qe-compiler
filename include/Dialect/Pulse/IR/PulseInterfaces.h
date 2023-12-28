//===- PulseInterfaces.h - Pulse dialect Interfaces -*- C++ -*-===============//
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
///  Interfaces for the Pulse dialect
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_INTERFACES_H
#define PULSE_INTERFACES_H

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/Support/Error.h"

//===----------------------------------------------------------------------===//
// Operation Interface Types
//===----------------------------------------------------------------------===//

namespace mlir::pulse::interfaces_impl {

//===----------------------------------------------------------------------===//
// PulseOpSchedulingInterface
//===----------------------------------------------------------------------===//

llvm::Optional<int64_t> getTimepoint(mlir::Operation *op);
void setTimepoint(mlir::Operation *op, int64_t timepoint);
llvm::Optional<uint64_t> getSetupLatency(mlir::Operation *op);
void setSetupLatency(mlir::Operation *op, uint64_t setupLatency);
llvm::Expected<uint64_t> getDuration(mlir::Operation *op,
                                     mlir::Operation *callSequenceOp = nullptr);
llvm::Expected<mlir::ArrayAttr> getPorts(mlir::Operation *op);
void setDuration(mlir::Operation *op, uint64_t duration);

} // namespace mlir::pulse::interfaces_impl

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseInterfaces.h.inc"

#endif // PULSE_INTERFACES_H

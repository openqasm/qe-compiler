//===- QUIRInterfaces.h - QUIR dialect Interfaces -*- C++ -*-=================//
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
///  Interfaces for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUIRINTERFACES_H
#define QUIR_QUIRINTERFACES_H

#include "mlir/IR/OpDefinition.h"

#include <set>

//===----------------------------------------------------------------------===//
// Operation Interface Types
//===----------------------------------------------------------------------===//

namespace mlir::quir::interfaces_impl {

//===----------------------------------------------------------------------===//
// QubitOpInterface
//===----------------------------------------------------------------------===//

/// Returns the nested qubits operated on within the operation.
std::set<uint32_t> getOperatedQubits(mlir::Operation *op,
                                     bool ignoreSelf = false);

/// Get the next (lexographically) Qubit operation implementing this interface
llvm::Optional<Operation *> getNextQubitOp(Operation *op);

/// @brief Get qubits that are shared between the two operations
std::set<uint32_t> getSharedQubits(std::set<uint32_t> &first,
                                   std::set<uint32_t> &second);

/// @brief Get qubits the union of the two qubit sets.
std::set<uint32_t> getUnionQubits(std::set<uint32_t> &first,
                                  std::set<uint32_t> &second);

/// @brief Get qubits that are shared between the two operations
std::set<uint32_t> getSharedQubits(Operation *first, Operation *second);

/// @brief This operation shares qubits with another
bool opsShareQubits(Operation *first, Operation *second);

/// @brief Check if the qubit sets overlap
bool qubitSetsOverlap(std::set<uint32_t> &first, std::set<uint32_t> &second);

/// @brief Get the qubits between two operations. Not including the operations
/// themselves
std::set<uint32_t> getQubitsBetweenOperations(mlir::Operation *first,
                                              mlir::Operation *second);

} // namespace mlir::quir::interfaces_impl

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRInterfaces.h.inc"

#endif // QUIR_QUIRINTERFACES_H

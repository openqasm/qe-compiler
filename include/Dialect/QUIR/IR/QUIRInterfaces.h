//===- QUIRInterfaces.h - QUIR dialect Interfaces -*- C++ -*-=================//
//
// (C) Copyright IBM 2022, 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

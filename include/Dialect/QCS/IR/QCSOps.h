//===- QCSOps.h - Quantum Control System dialect ops ------------*- C++ -*-===//
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
/// This file declares the operations in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QCS_QCSOPS_H_
#define DIALECT_QCS_QCSOPS_H_

#include <unordered_map>

// TODO: move necessary components to `QCS`
#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"

#include "Dialect/QCS/IR/QCSTypes.h"

#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "Dialect/QCS/IR/QCSOps.h.inc"

#endif // DIALECT_QCS_QCSOPS_H_

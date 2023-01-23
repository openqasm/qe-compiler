//===- QCSOps.h - Quantum Control System dialect ops ------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

// TODO: move necessary components to `QCS`
#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"

#include "Dialect/QCS/IR/QCSTypes.h"

#define GET_OP_CLASSES
#include "Dialect/QCS/IR/QCSOps.h.inc"

#endif // DIALECT_QCS_QCSOPS_H_

//===- QuSysOps.h - System dialect ops -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file declares the operations in the Quantum System dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QUSYS_QUSYSOPS_H_
#define DIALECT_QUSYS_QUSYSOPS_H_

// TODO: move necessary components to `QuSys`
#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"

#include "Dialect/QuSys/IR/QuSysTypes.h"

#define GET_OP_CLASSES
#include "Dialect/QuSys/IR/QuSysOps.h.inc"

#endif // DIALECT_QUSYS_QUSYSOPS_H_

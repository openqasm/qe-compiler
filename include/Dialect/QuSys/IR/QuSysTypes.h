//===- SystemTypes.h - System dialect types ---------------*- C++ -*-=========//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file declares the types in the Quantum System dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QUSYS_QUSYSTYPES_H_
#define DIALECT_QUSYS_QUSYSTYPES_H_

// TODO: Temporary, until constraints between `OpenQASM3`, `QUIR`, `Pulse`, and
// `System` dialects are ironed out.
#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir::qusys {} // namespace mlir::qusys

#endif // DIALECT_QUSYS_QUSYSTYPES_H_

//===- QCSTypes.h - Quantum Control System dialect types --*- C++ -*-=========//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file declares the types in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QCS_QCSTYPES_H_
#define DIALECT_QCS_QCSTYPES_H_

// TODO: Temporary, until constraints between `OpenQASM3`, `QUIR`, `Pulse`, and
// `System` dialects are ironed out.
#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir::qcs {} // namespace mlir::qcs

#endif // DIALECT_QCS_QCSTYPES_H_

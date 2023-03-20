//===- QCSTypes.h - Quantum Control System dialect types --*- C++ -*-=========//
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

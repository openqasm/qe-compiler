//===- QCSAttributes.h - QCS dialect attributes -----------------*- C++ -*-===//
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
///  This file declares Quantum Control System dialect specific attributes.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QCS_QCSATTRIBUTES_H_
#define DIALECT_QCS_QCSATTRIBUTES_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::qcs {
static inline llvm::StringRef getShotLoopAttrName() { return "qcs.shot_loop"; }
static inline llvm::StringRef getNumShotsAttrName() { return "qcs.num_shots"; }
} // namespace mlir::qcs

#endif // DIALECT_QCS_QCSATTRIBUTES_H_

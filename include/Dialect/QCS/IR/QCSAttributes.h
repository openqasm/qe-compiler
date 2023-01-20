//===- QCSAttributes.h - QCS dialect attributes -----------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

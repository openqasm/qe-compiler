//===- QuSysAttributes.h - QuSys dialect attributes -------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares Quantum System dialect specific attributes.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QUSYS_QUSYSATTRIBUTES_H_
#define DIALECT_QUSYS_QUSYSATTRIBUTES_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::qusys {
static inline llvm::StringRef getShotLoopAttrName() {
  return "qusys.shot_loop";
}
static inline llvm::StringRef getNumShotsAttrName() {
  return "qusys.num_shots";
}
} // namespace mlir::qusys

#endif // DIALECT_QUSYS_QUSYSATTRIBUTES_H_

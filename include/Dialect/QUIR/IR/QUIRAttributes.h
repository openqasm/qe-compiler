//===- QUIRAttributes.h - QUIR dialect attributes ---------------*- C++ -*-===//
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
///  Attributes for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUIRATTRIBUTES_H
#define QUIR_QUIRATTRIBUTES_H

#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"

#include "Dialect/QUIR/IR/QUIREnums.h.inc"

namespace mlir::quir {

static inline llvm::StringRef getInputParameterAttrName() {
  return "quir.inputParameter";
}
static inline llvm::StringRef getPhysicalIdAttrName() {
  return "quir.physicalId";
}
static inline llvm::StringRef getPhysicalIdsAttrName() {
  return "quir.physicalIds";
}
static inline llvm::StringRef getNoReportRuntimeAttrName() {
  return "quir.noReportRuntime";
}
static inline llvm::StringRef getNoReportUserResultAttrName() {
  return "quir.noReportUserResult";
}

static inline llvm::StringRef getAngleAttrName() { return "quir.angleValue"; }

static inline llvm::StringRef getDurationAttrName() {
  return "quir.durationValue";
}

/// Duration representation
// Question: Can this be represented with MLIR more naturally?
// TODO: This should be added to the DurationAttr
struct Duration {
  enum DurationUnit { dt, ns, us, ms, s };
  double duration;
  DurationUnit unit;

  /// Construct a Duration from a string
  static llvm::Expected<Duration> parseDuration(const std::string &durationStr);
  /// Convert duration to cycles. dt is in SI (seconds).
  Duration convertToCycles(double dt) const;
};

} // namespace mlir::quir

#define GET_ATTRDEF_CLASSES
#include "Dialect/QUIR/IR/QUIRAttributes.h.inc"

#endif // QUIR_QUIRATTRIBUTES_H

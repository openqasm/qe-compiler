//===- SystemAttributes.h - System dialect attributes -----*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  Attributes for the System dialect
///
//===----------------------------------------------------------------------===//

#ifndef SYSTEM_SYSTEMATTRIBUTES_H
#define SYSTEM_SYSTEMATTRIBUTES_H

#include "Dialect/System/IR/SystemTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/System/IR/SystemAttributes.h.inc"

namespace mlir::sys {} // namespace mlir::sys

#endif // SYSTEM_SYSTEMATTRIBUTES_H

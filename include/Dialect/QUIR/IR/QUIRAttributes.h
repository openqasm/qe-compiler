//===- QUIRAttributes.h - QUIR dialect attributes ---------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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

#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/QUIR/IR/QUIRAttributes.h.inc"

#endif // QUIR_QUIRATTRIBUTES_H

//===- QUIRCast.h - Convert cast op to Std ----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  This file exposes patterns for converting the QUIR cast op
///  to the std dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIRTOSTD_QUIRCAST_H
#define QUIRTOSTD_QUIRCAST_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir::quir {

void populateQUIRCastPatterns(RewritePatternSet &patterns,
                              mlir::TypeConverter &typeConverter);

} // namespace mlir::quir

#endif // QUIRTOSTD_QUIRCAST_H

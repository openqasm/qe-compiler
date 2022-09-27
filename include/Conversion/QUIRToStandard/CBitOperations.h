//===- CBitOperations.h -----------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file exposes patterns for lowering operations on cbits.
///
//===----------------------------------------------------------------------===//

#ifndef QUIRTOSTD_CBITOPERATIONS_H
#define QUIRTOSTD_CBITOPERATIONS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir::quir {

void populateCBitOperationsPatterns(RewritePatternSet &patterns,
                                    mlir::TypeConverter &typeConverter);

}; // namespace mlir::quir

#endif // QUIRTOSTD_CBITOPERATIONS_H

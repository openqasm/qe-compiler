//===- OQ3ToStandard.h - OpenQASM 3 to Standard patterns --------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file implements patterns to convert OpenQASM 3 to Standard dialect.
///
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_OQ3TOSTANDARD_OQ3TOSTANDARD_H_
#define CONVERSION_OQ3TOSTANDARD_OQ3TOSTANDARD_H_

#include "Dialect/OQ3/IR/OQ3Types.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::oq3 {
// Appends to a pattern list additional patterns for translating OpenQASM 3
// ops to Standard ops.
void populateOQ3ToStandardConversionPatterns(
    RewritePatternSet &patterns, bool includeBitmapOperationPatterns = true);
}; // namespace mlir::oq3

#endif // CONVERSION_OQ3TOSTANDARD_OQ3TOSTANDARD_H_

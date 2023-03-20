//===- CBitOperations.h -----------------------------------------*- C++ -*-===//
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
                                    mlir::TypeConverter &typeConverter,
                                    bool includeBitmapOperationPatterns = true);

}; // namespace mlir::quir

#endif // QUIRTOSTD_CBITOPERATIONS_H

//===- VariablesToGobalMemRefConversion.h -----------------------*- C++ -*-===//
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
/// This file exposes patterns for lowering QUIR variable declarations,
/// variable use, and variable assignments to std. For that purpose, it
/// introduces a global variable for each QUIR variable declaration via a
/// GlobalMemRef. All variable references and assignments are converted into
/// load and store op against the global memrefs.
///
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir::quir {

void populateVariableToGlobalMemRefConversionPatterns(
    RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    bool externalizeOutputVariables);

}; // namespace mlir::quir

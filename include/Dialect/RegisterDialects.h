//===- OperationUtils.h -----------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  Utility functions related to MLIR Dialect handling.
///
//===----------------------------------------------------------------------===//

#ifndef UTILS_DIALECTUTILS_H
#define UTILS_DIALECTUTILS_H

#include "mlir/InitAllDialects.h"

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"

namespace qssc::dialect {

/// Register all qss-compiler dialects returning a dialect registry
inline mlir::DialectRegistry registerDialects() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::quir::QUIRDialect, mlir::pulse::PulseDialect,
                  mlir::qcs::QCSDialect>();
  return registry;
}
} // namespace qssc::dialect

#endif

//===- OperationUtils.h -----------------------------------------*- C++ -*-===//
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
///  Utility functions related to MLIR Dialect handling.
///
//===----------------------------------------------------------------------===//

#ifndef UTILS_DIALECTUTILS_H
#define UTILS_DIALECTUTILS_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"

#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"

namespace qssc::dialect {

/// Register all qss-compiler dialects returning a dialect registry
inline void registerDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  registry.insert<mlir::oq3::OQ3Dialect, mlir::quir::QUIRDialect,
                  mlir::pulse::PulseDialect, mlir::qcs::QCSDialect>();
}
} // namespace qssc::dialect

#endif

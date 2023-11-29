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

#ifndef REGISTER_PASSES_H
#define REGISTER_PASSES_H

#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/OQ3/Transforms/Passes.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/Transforms/Passes.h"
#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QCS/Utils/ParameterInitialValueAnalysis.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/Transforms/Passes.h"
#include "HAL/PassRegistration.h"

#include "mlir/InitAllPasses.h"

namespace qssc::dialect {

/// Register all qss-compiler passes
inline llvm::Error registerPasses() {
  // TODO: Register standalone passes here.
  llvm::Error err = llvm::Error::success();
  mlir::oq3::registerOQ3Passes();
  mlir::oq3::registerOQ3PassPipeline();
  mlir::qcs::registerQCSPasses();
  mlir::quir::registerQuirPasses();
  mlir::quir::registerQuirPassPipeline();
  mlir::pulse::registerPulsePasses();
  mlir::pulse::registerPulsePassPipeline();
  mlir::registerConversionPasses();

  err = llvm::joinErrors(std::move(err), qssc::hal::registerTargetPasses());
  err = llvm::joinErrors(std::move(err), qssc::hal::registerTargetPipelines());

  mlir::registerAllPasses();
  return err;
}

} // namespace qssc::dialect

#endif // REGISTER_PASSES_H

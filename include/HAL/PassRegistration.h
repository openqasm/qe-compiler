//===- PassRegistration.h - Top-level pass registration ---------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file declares the top-level functions for registering passes and
//  pipelines
//
//===----------------------------------------------------------------------===//
#ifndef PASSREGISTRATION_H
#define PASSREGISTRATION_H

#include "llvm/Support/Error.h"

namespace qssc::hal {
/// Register all registered MLIR passes
/// for the registered Targets with the
/// QSSC system.
llvm::Error registerTargetPasses();
/// Register all registered MLIR pass pipelines
/// for the registered Targets with the
/// QSSC system.
llvm::Error registerTargetPipelines();
} // namespace qssc::hal

#endif // PASSREGISTRATION_H

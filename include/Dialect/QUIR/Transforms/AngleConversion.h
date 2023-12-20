//===- AngleConversion.h - Convert CallGateOp Angles  ----------*- C++ -*-===//
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
///  This file declares the pass for converting angles in CallGateOp
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_ANGLE_CONVERSION_H
#define QUIR_ANGLE_CONVERSION_H

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_map>

namespace mlir::quir {
struct QUIRAngleConversionPass
    : public PassWrapper<QUIRAngleConversionPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName =
      "QUIR Angle Conversion Pass (" + getArgument().str() + ")";

private:
  std::unordered_map<std::string, FuncOp> functionOps;
}; // struct QUIRAngleConversionPass

} // end namespace mlir::quir

#endif // QUIR_ANGLE_CONVERSION_H

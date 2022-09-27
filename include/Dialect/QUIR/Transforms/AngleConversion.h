//===- AngleConversion.h - Convert CallGateOp Angles  ----------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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

namespace mlir::quir {
struct QUIRAngleConversionPass
    : public PassWrapper<QUIRAngleConversionPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct QUIRAngleConversionPass

} // end namespace mlir::quir

#endif // QUIR_ANGLE_CONVERSION_H

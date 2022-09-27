//===- InlineRegion.h - Inlining all dialects -------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass for inlining all dialects
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_INLINE_REGION_H
#define PULSE_INLINE_REGION_H

#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

class InlineRegionPass
    : public PassWrapper<InlineRegionPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};

} // namespace mlir::pulse
#endif // PULSE_INLINE_REGION_H

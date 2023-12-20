//===- InlineRegion.h - Inlining all dialects -------------------*- C++ -*-===//
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
  llvm::StringRef getName() const override;
  std::string passName = 
      "Inline Region Pass (" + getArgument().str() + ")";
};

} // namespace mlir::pulse
#endif // PULSE_INLINE_REGION_H

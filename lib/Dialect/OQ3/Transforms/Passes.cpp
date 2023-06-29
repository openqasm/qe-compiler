//===- Passes.cpp - OQ3 Passes ----------------------------------*- C++ -*-===//
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

#include "Dialect/OQ3/Transforms/Passes.h"

#include "Dialect/OQ3/Transforms/LimitCBitWidth.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::oq3 {

void oq3PassPipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<LimitCBitWidthPass>());
}

void registerOQ3Passes() {
  //===----------------------------------------------------------------------===//
  // Transform Passes
  //===----------------------------------------------------------------------===//
  PassRegistration<LimitCBitWidthPass>();
}

void registerOQ3PassPipeline() {
  PassPipelineRegistration<> pipeline(
      "oq3Opt", "Enable OQ3-specific optimizations", oq3PassPipelineBuilder);
}
} // end namespace mlir::oq3

//===- Passes.cpp - Pulse Passes --------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/Passes.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::pulse {

void pulsePassPipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<SystemCreationPass>());
  pm.addPass(std::make_unique<QUIRToPulsePass>());
}

void registerPulsePasses() {
  PassRegistration<SystemCreationPass>();
  PassRegistration<SystemPlotPass>();
  PassRegistration<QUIRToPulsePass>();
  PassRegistration<PortGroupPrunePass>();
  PassRegistration<SlicePortPass>();
  PassRegistration<InlineRegionPass>();
}

void registerPulsePassPipeline() {
  PassPipelineRegistration<> pipeline("pulseOpt",
                                      "Enable Pulse IR specific optimizations",
                                      pulse::pulsePassPipelineBuilder);
}

} // namespace mlir::pulse

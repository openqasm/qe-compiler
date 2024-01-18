//===- Passes.cpp - Pulse Passes --------------------------------*- C++ -*-===//
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
///  This file implements functions for registering the core Pulse IR passes.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/Passes.h"

#include "Conversion/QUIRToPulse/LoadPulseCals.h"
#include "Conversion/QUIRToPulse/QUIRToPulse.h"

#include "Dialect/Pulse/Transforms/ClassicalOnlyDetection.h"
#include "Dialect/Pulse/Transforms/LabelPlayOpDurations.h"
#include "Dialect/Pulse/Transforms/MergeDelays.h"
#include "Dialect/Pulse/Transforms/RemoveUnusedArguments.h"
#include "Dialect/Pulse/Transforms/SchedulePort.h"

#include "Dialect/Pulse/Transforms/Scheduling.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::pulse {

void pulsePassPipelineBuilder(OpPassManager &pm) {}

void registerPulsePasses() {
  PassRegistration<LabelPlayOpDurationsPass>();
  PassRegistration<LoadPulseCalsPass>();
  PassRegistration<QUIRToPulsePass>();
  PassRegistration<MergeDelayPass>();
  PassRegistration<RemoveUnusedArgumentsPass>();
  PassRegistration<SchedulePortPass>();
  PassRegistration<QuantumCircuitPulseSchedulingPass>();
  PassRegistration<ClassicalOnlyDetectionPass>();
}

void registerPulsePassPipeline() {
  PassPipelineRegistration<> const pipeline(
      "pulseOpt", "Enable Pulse IR specific optimizations",
      pulse::pulsePassPipelineBuilder);
}

} // namespace mlir::pulse

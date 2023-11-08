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

void pulsePassPipelineBuilder(OpPassManager &pm) {}

void registerPulsePasses() {
  PassRegistration<LoadPulseCalsPass>();
  PassRegistration<QUIRToPulsePass>();
  PassRegistration<MergeDelayPass>();
  PassRegistration<RemoveUnusedArgumentsPass>();
  PassRegistration<SchedulePortPass>();
  PassRegistration<SchedulingPulseSequencesPass>();
  PassRegistration<ClassicalOnlyDetectionPass>();
}

void registerPulsePassPipeline() {
  PassPipelineRegistration<> pipeline("pulseOpt",
                                      "Enable Pulse IR specific optimizations",
                                      pulse::pulsePassPipelineBuilder);
}

} // namespace mlir::pulse

//===- Passes.h - Pulse Passes ----------------------------------*- C++ -*-===//
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
///  This file declares functions for registering the core Pulse IR passes.
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_PULSEPASSES_H
#define PULSE_PULSEPASSES_H

#include "Conversion/QUIRToPulse/LoadPulseCals.h"
#include "Conversion/QUIRToPulse/QUIRToPulse.h"
#include "Dialect/Pulse/Transforms/ClassicalOnlyDetection.h"
#include "Dialect/Pulse/Transforms/LabelPlayOpDurations.h"
#include "Dialect/Pulse/Transforms/MergeDelays.h"
#include "Dialect/Pulse/Transforms/RemoveUnusedArguments.h"
#include "Dialect/Pulse/Transforms/SchedulePort.h"
#include "Dialect/Pulse/Transforms/Scheduling.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::pulse {
void registerPulsePasses();       // individual command line passes
void registerPulsePassPipeline(); // pass pipeline (ordered list of passes)
void pulsePassPipelineBuilder(OpPassManager &pm);
} // namespace mlir::pulse

#endif // PULSE_PULSEPASSES_H

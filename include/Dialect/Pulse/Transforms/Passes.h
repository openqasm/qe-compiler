//===- Passes.h - Pulse Passes ----------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
///  This file declares functions for registering the core Pulse IR passes.
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_PULSEPASSES_H
#define PULSE_PULSEPASSES_H

#include "Conversion/QUIRToPulse/QUIRToPulse.h"
#include "Dialect/Pulse/Transforms/ClassicalOnlyDetection.h"
#include "Dialect/Pulse/Transforms/MergeDelays.h"
#include "Dialect/Pulse/Transforms/RemoveUnusedArguments.h"
#include "Dialect/Pulse/Transforms/SchedulePort.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::pulse {
void registerPulsePasses();       // individual command line passes
void registerPulsePassPipeline(); // pass pipeline (ordered list of passes)
void pulsePassPipelineBuilder(OpPassManager &pm);
} // namespace mlir::pulse

#endif // PULSE_PULSEPASSES_H

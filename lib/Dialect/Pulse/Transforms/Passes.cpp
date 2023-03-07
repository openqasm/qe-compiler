//===- Passes.cpp - Pulse Passes --------------------------------*- C++ -*-===//
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
  PassRegistration<QUIRToPulsePass>();
  PassRegistration<MergeDelayPass>();
  PassRegistration<RemoveUnusedArgumentsPass>();
  PassRegistration<SchedulePortPass>();
  PassRegistration<ClassicalOnlyDetectionPass>();
}

void registerPulsePassPipeline() {
  PassPipelineRegistration<> pipeline("pulseOpt",
                                      "Enable Pulse IR specific optimizations",
                                      pulse::pulsePassPipelineBuilder);
}

} // namespace mlir::pulse

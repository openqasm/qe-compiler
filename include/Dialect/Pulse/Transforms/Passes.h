//===- Passes.h - Pulse Passes ----------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef PULSE_PULSEPASSES_H
#define PULSE_PULSEPASSES_H

#include "Conversion/QUIRToPulse/QUIRToPulse.h"
#include "Dialect/Pulse/Transforms/InlineRegion.h"
#include "Dialect/Pulse/Transforms/PortGroupPrune.h"
#include "Dialect/Pulse/Transforms/SlicePorts.h"
#include "Dialect/Pulse/Transforms/SystemCreation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::pulse {
void registerPulsePasses();       // individual command line passes
void registerPulsePassPipeline(); // pass pipeline (ordered list of passes)
} // namespace mlir::pulse

#endif // PULSE_PULSEPASSES_H

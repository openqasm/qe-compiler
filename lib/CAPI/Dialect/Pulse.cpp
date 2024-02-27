//===- Pulse.cpp - Pulse dialect CAPI registration --------------*- C++ -*-===//
//
// (C) Copyright IBM 2024.
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
///  This file implements the Pulse dialect registration of the CAPI
///
//===----------------------------------------------------------------------===//

#include "qss-c/Dialect/Pulse.h"

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseTypes.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Pulse, pulse, pulse::PulseDialect)

//===---------------------------------------------------------------------===//
// CaptureType
//===---------------------------------------------------------------------===//

bool pulseTypeIsACaptureType(MlirType type) {
  return unwrap(type).isa<pulse::CaptureType>();
}

MlirType pulseCaptureTypeGet(MlirContext ctx) {
  return wrap(pulse::CaptureType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// FrameType
//===---------------------------------------------------------------------===//

bool pulseTypeIsAFrameType(MlirType type) {
  return unwrap(type).isa<pulse::FrameType>();
}

MlirType pulseFrameTypeGet(MlirContext ctx) {
  return wrap(pulse::FrameType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// KernelType
//===---------------------------------------------------------------------===//

bool pulseTypeIsAKernelType(MlirType type) {
  return unwrap(type).isa<pulse::KernelType>();
}

MlirType pulseKernelTypeGet(MlirContext ctx) {
  return wrap(pulse::KernelType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// MixedFrameType
//===---------------------------------------------------------------------===//

bool pulseTypeIsAMixedFrameType(MlirType type) {
  return unwrap(type).isa<pulse::MixedFrameType>();
}

MlirType pulseMixedFrameTypeGet(MlirContext ctx) {
  return wrap(pulse::MixedFrameType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// PortType
//===---------------------------------------------------------------------===//

bool pulseTypeIsAPortType(MlirType type) {
  return unwrap(type).isa<pulse::PortType>();
}

MlirType pulsePortTypeGet(MlirContext ctx) {
  return wrap(pulse::PortType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// WaveformType
//===---------------------------------------------------------------------===//

bool pulseTypeIsAWaveformType(MlirType type) {
  return unwrap(type).isa<pulse::WaveformType>();
}

MlirType pulseWaveformTypeGet(MlirContext ctx) {
  return wrap(pulse::WaveformType::get(unwrap(ctx)));
}

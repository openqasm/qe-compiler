//===- PDL.cpp - C Interface for PDL dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

//===-- pulse.h - C API for pulse Dialect -------------------------*- C -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef C_DIALECT_PULSE_H
#define C_DIALECT_PULSE_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PULSE, pulse);

//===---------------------------------------------------------------------===//
// CaptureType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool pulseTypeIsACaptureType(MlirType type);

MLIR_CAPI_EXPORTED MlirType pulseCaptureTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// FrameType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool pulseTypeIsAFrameType(MlirType type);

MLIR_CAPI_EXPORTED MlirType pulseFrameTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// KernelType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool pulseTypeIsAKernelType(MlirType type);

MLIR_CAPI_EXPORTED MlirType pulseKernelTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// MixedFrameType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool pulseTypeIsAMixedFrameType(MlirType type);

MLIR_CAPI_EXPORTED MlirType pulseMixedFrameTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// PortType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool pulseTypeIsAPortType(MlirType type);

MLIR_CAPI_EXPORTED MlirType pulsePortTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// WaveformType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool pulseTypeIsAWaveformType(MlirType type);

MLIR_CAPI_EXPORTED MlirType pulseWaveformTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // C_DIALECT_PULSE_H

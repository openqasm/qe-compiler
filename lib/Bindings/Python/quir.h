//===-- pulse.h - C API for pulse Dialect -------------------------*- C -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef C_DIALECT_QUIR_H
#define C_DIALECT_QUIR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QUIR, quir);

//===---------------------------------------------------------------------===//
// AngleType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool quirTypeIsAAngleType(MlirType type);

MLIR_CAPI_EXPORTED MlirType quirAngleTypeGet(MlirContext ctx, unsigned width);

//===---------------------------------------------------------------------===//
// DurationType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool quirTypeIsADurationType(MlirType type);

MLIR_CAPI_EXPORTED MlirType quirDurationTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // C_DIALECT_QUIR_H

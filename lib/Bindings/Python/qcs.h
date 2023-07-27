//===-- pulse.h - C API for pulse Dialect -------------------------*- C -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef C_DIALECT_QCS_H
#define C_DIALECT_QCS_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QCS, qcs);

#ifdef __cplusplus
}
#endif

#endif // C_DIALECT_QUIR_H

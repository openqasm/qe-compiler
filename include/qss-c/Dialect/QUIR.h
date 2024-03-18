//===- QUIR.h - QUIR Dialect for C ------------------------------*- C++ -*-===//
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
///  This file declares the c interface for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef C_DIALECT_QUIR_H
#define C_DIALECT_QUIR_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

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

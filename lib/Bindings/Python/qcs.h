//===- qcs.h - QCS Dialect python CAPI registration -------------*- C++ -*-===//
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
///  This file declares the CAPI python bindings for the QCS dialect
///
//===----------------------------------------------------------------------===//

#ifndef C_DIALECT_QCS_H
#define C_DIALECT_QCS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QCS, qcs);

#ifdef __cplusplus
}
#endif

#endif // C_DIALECT_QUIR_H

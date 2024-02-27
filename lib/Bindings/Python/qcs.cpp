//===- qcs.cpp - QCS Dialect python CAPI registration -----------*- C++ -*-===//
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
///  This file implements the CAPI python bindings for the QCS dialect
///
//===----------------------------------------------------------------------===//

#include "qss-c/Dialect/QCS.h"

#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QCS/IR/QCSTypes.h"

#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QCS, qcs, qcs::QCSDialect)

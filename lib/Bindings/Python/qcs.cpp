//===- PDL.cpp - C Interface for PDL dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "qcs.h"

#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QCS/IR/QCSTypes.h"

#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QCS, qcs, qcs::QCSDialect)


//===- PDL.cpp - C Interface for PDL dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "qss-c/Dialect/OQ3.h"

#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/OQ3/IR/OQ3Types.h"

#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(OQ3, oq3, oq3::OQ3Dialect)


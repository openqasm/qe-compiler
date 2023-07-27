//===- PDL.cpp - C Interface for PDL dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "quir.h"

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QUIR, quir, quir::QUIRDialect)

//===---------------------------------------------------------------------===//
// AngleType
//===---------------------------------------------------------------------===//

bool quirTypeIsAAngleType(MlirType type) {
  return unwrap(type).isa<quir::AngleType>();
}

MlirType quirAngleTypeGet(MlirContext ctx, unsigned width) {

  // TODO: NEED TO ADD WIDTH PARAMETER
  return wrap(quir::AngleType::get(unwrap(ctx), width));
}

//===---------------------------------------------------------------------===//
// DurationType
//===---------------------------------------------------------------------===//

bool quirTypeIsADurationType(MlirType type) {
  return unwrap(type).isa<quir::DurationType>();
}

MlirType quirDurationTypeGet(MlirContext ctx) {

  // TODO: NEED TO ADD WIDTH PARAMETER
  return wrap(quir::DurationType::get(unwrap(ctx)));
}

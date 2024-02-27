//===- quir.cpp - QUIR Dialect python CAPI registration ---------*- C++ -*-===//
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
///  This file implements the CAPI python bindings for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#include "qss-c/Dialect/QUIR.h"

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
  return wrap(quir::DurationType::get(unwrap(ctx), mlir::quir::TimeUnits::dt));
}

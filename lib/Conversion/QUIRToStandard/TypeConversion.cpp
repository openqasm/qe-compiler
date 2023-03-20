//===- TypeConversion.cpp - Convert QUIR types to Std -----*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
///  This file implements common utilities for converting QUIR to std
///
//===----------------------------------------------------------------------===//

#include "Conversion/QUIRToStandard/TypeConversion.h"
#include "Dialect/QUIR/IR/QUIROps.h"

namespace mlir {

namespace {
Optional<Type> convertCbitType(quir::CBitType t) {

  if (t.getWidth() <= 64)
    return IntegerType::get(t.getContext(), t.getWidth());

  return llvm::None;
}

Optional<Type> legalizeIndexType(mlir::IndexType t) { return t; }
} // anonymous namespace

QuirTypeConverter::QuirTypeConverter() {
  addConversion(convertAngleType);
  addSourceMaterialization(angleSourceMaterialization);

  addConversion(convertCbitType);
  addConversion(legalizeIndexType);
}

Optional<Type> QuirTypeConverter::convertAngleType(Type t) {

  auto *context = t.getContext();
  if (auto angleType = t.dyn_cast<quir::AngleType>()) {
    auto width = angleType.getWidth();

    if (!width.hasValue()) {
      llvm::errs() << "Cannot lower an angle with no width!\n";
      return {};
    }
    if (width > 31)
      return IntegerType::get(context, 64);
    return IntegerType::get(context, 32);
  }
  if (auto intType = t.dyn_cast<IntegerType>()) {
    // MUST return the converted type as itself to mark legal
    // for function types in func defs and calls
    return intType;
  }
  return llvm::None;
} // convertAngleType

Optional<Value> QuirTypeConverter::angleSourceMaterialization(
    OpBuilder &builder, quir::AngleType aType, ValueRange valRange,
    Location loc) {
  for (Value val : valRange) {
    auto castOp = builder.create<quir::CastOp>(loc, aType, val);
    return castOp.out();
  } // for val : valRange
  return llvm::None;
} // angleSourceMaterialization
} // namespace mlir

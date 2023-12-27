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
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace mlir {

namespace {
std::optional<Type> convertCBitType(quir::CBitType t) {

  if (t.getWidth() <= 64)
    return IntegerType::get(t.getContext(), t.getWidth());

  return std::nullopt;
}

std::optional<Type> legalizeIndexType(mlir::IndexType t) { return t; }
} // anonymous namespace

QuirTypeConverter::QuirTypeConverter() {
  addConversion(convertAngleType);
  addSourceMaterialization(angleSourceMaterialization);

  addConversion(convertCBitType);
  addConversion(legalizeIndexType);
}

std::optional<Type> QuirTypeConverter::convertAngleType(Type t) {

  auto *context = t.getContext();
  if (auto angleType = t.dyn_cast<quir::AngleType>()) {
    auto width = angleType.getWidth();

    if (!width.has_value()) {
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
  if (auto floatType = t.dyn_cast<FloatType>()) {
    // MUST return the converted type as itself to mark legal
    // for function types in func defs and calls
    return floatType;
  }
  return std::nullopt;
} // convertAngleType

std::optional<Value> QuirTypeConverter::angleSourceMaterialization(
    OpBuilder &builder, quir::AngleType aType, ValueRange valRange,
    Location loc) {
  for (Value const val : valRange) {
    auto castOp = builder.create<oq3::CastOp>(loc, aType, val);
    return castOp.getOut();
  } // for val : valRange
  return std::nullopt;
} // angleSourceMaterialization
} // namespace mlir

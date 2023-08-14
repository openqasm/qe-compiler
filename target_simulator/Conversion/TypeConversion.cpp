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

#include "Conversion/TypeConversion.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIROps.h"

namespace mlir {

namespace {
Optional<Type> convertCBitType(quir::CBitType t) {

  if (t.getWidth() <= 64)
    return IntegerType::get(t.getContext(), t.getWidth());

  return llvm::None;
}

Optional<Type> legalizeIndexType(mlir::IndexType t) { return t; }
} // anonymous namespace

AerTypeConverter::AerTypeConverter() {
  addConversion(convertQubitType);
  addConversion(convertAngleType);
  addSourceMaterialization(qubitSourceMaterialization);
  addSourceMaterialization(cBitSourceMaterialization);
  addSourceMaterialization(angleSourceMaterialization);
  addSourceMaterialization(durationSourceMaterialization);

  addConversion(convertCBitType);
  addConversion(legalizeIndexType);
}

Optional<Type> AerTypeConverter::convertQubitType(Type t) {
  auto *context = t.getContext();
  return IntegerType::get(context, 64);
}

Optional<Type> AerTypeConverter::convertAngleType(Type t) {

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

Optional<Value> AerTypeConverter::qubitSourceMaterialization(
    OpBuilder &builder, quir::QubitType qType, ValueRange values,
    Location loc) {
    for(Value val : values) {
        return val;
    }
    return llvm::None;
}

// TODO
Optional<Value> AerTypeConverter::cBitSourceMaterialization(
    OpBuilder &builder, quir::CBitType qType, ValueRange values,
    Location loc) {
    for(Value val : values) {
        return val;
    }
    return llvm::None;
}

Optional<Value> AerTypeConverter::angleSourceMaterialization(
    OpBuilder &builder, quir::AngleType aType, ValueRange valRange,
    Location loc) {
  for (Value val : valRange) {
    auto castOp = builder.create<oq3::CastOp>(loc, aType, val);
    return castOp.out();
  } // for val : valRange
  return llvm::None;
} // angleSourceMaterialization

// TODO
Optional<Value> AerTypeConverter::durationSourceMaterialization(
    OpBuilder &builder, quir::DurationType dType, ValueRange valRange,
    Location loc) {
  for (Value val : valRange) {
    return val;
  }
  return llvm::None;
}
} // namespace mlir

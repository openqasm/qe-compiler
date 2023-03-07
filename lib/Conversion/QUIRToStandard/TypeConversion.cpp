//===- TypeConversion.cpp - Convert QUIR types to Std -----*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
//  This file implements common utilities for converting QUIR to std
//
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

//===- TypeConversion.h - Convert QUIR types to Std -----*- C++ -*-===//
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
//
//  This file declares common utilities for converting QUIR to std
//
//===----------------------------------------------------------------------===//

#ifndef AERTOSTD_TYPECONVERSION__H
#define AERTOSTD_TYPECONVERSION__H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir { // TODO: change namespace

struct AerTypeConverter : public TypeConverter {
  using mlir::TypeConverter::TypeConverter;

  AerTypeConverter();

  static Optional<Type> convertQubitType(Type t);
  static Optional<Type> convertAngleType(Type t);
  static Optional<Type> convertDurationType(Type t);

  static Optional<Value> qubitSourceMaterialization(
      OpBuilder &builder, quir::QubitType qType,
      ValueRange valRange, Location loc);
  static Optional<Value> cBitSourceMaterialization(
      OpBuilder &builder, quir::CBitType qType,
      ValueRange valRange, Location loc);
  static Optional<Value> angleSourceMaterialization(
      OpBuilder &builder, quir::AngleType aType,
      ValueRange valRange, Location loc);
  static Optional<Value> durationSourceMaterialization(
      OpBuilder &builder, quir::DurationType dType,
      ValueRange valRange, Location loc);

}; // struct AerTypeConverter

} // end namespace mlir

#endif // AERTOSTD_TYPECONVERSION__H

//===- TypeConversion.h - Convert QUIR types to Std -------------*- C++ -*-===//
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

namespace qssc::targets::simulator::aer {

struct AerTypeConverter : public mlir::TypeConverter {
  using mlir::TypeConverter::TypeConverter;

  AerTypeConverter();

  static mlir::Optional<mlir::Type> convertQubitType(mlir::Type t);
  static mlir::Optional<mlir::Type> convertAngleType(mlir::Type t);
  static mlir::Optional<mlir::Type> convertDurationType(mlir::Type t);

  static mlir::Optional<mlir::Value>
  qubitSourceMaterialization(mlir::OpBuilder &builder,
                             mlir::quir::QubitType qType,
                             mlir::ValueRange valRange, mlir::Location loc);
  static mlir::Optional<mlir::Value>
  angleSourceMaterialization(mlir::OpBuilder &builder,
                             mlir::quir::AngleType aType,
                             mlir::ValueRange valRange, mlir::Location loc);
  static mlir::Optional<mlir::Value>
  durationSourceMaterialization(mlir::OpBuilder &builder,
                                mlir::quir::DurationType dType,
                                mlir::ValueRange valRange, mlir::Location loc);

}; // struct AerTypeConverter

} // namespace qssc::targets::simulator::aer

#endif // AERTOSTD_TYPECONVERSION__H

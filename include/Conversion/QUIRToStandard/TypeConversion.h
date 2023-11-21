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

#ifndef QUIRTOSTD_TYPECONVERSION__H
#define QUIRTOSTD_TYPECONVERSION__H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir {

struct QuirTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  QuirTypeConverter();

  static std::optional<Type> convertAngleType(Type t); // convertAngleType

  static std::optional<Value> angleSourceMaterialization(OpBuilder &builder,
                                                    quir::AngleType aType,
                                                    ValueRange valRange,
                                                    Location loc);

}; // struct QuirTypeConverter

} // end namespace mlir

#endif // QUIRTOSTD_TYPECONVERSION__H

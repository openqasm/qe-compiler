//===- QUIREnums.h - QUIR dialect enums -------------------------*- C++ -*-===//
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

#ifndef QUIR_QUIRENUMS_H
#define QUIR_QUIRENUMS_H


#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#define GET_ENUM_CLASSES
#include "Dialect/QUIR/IR/QUIREnums.h.inc"


namespace mlir {

AsmPrinter &operator<<(AsmPrinter &printer, quir::TimeUnits param) {
  printer << stringifyEnum(param);
  return printer;
}

template <> struct mlir::FieldParser<quir::TimeUnits> {
  static FailureOr<quir::TimeUnits> parse(AsmParser &parser) {
    std::string unit;
    if (parser.parseString(&unit))
      return failure();

    if (auto unitEnum = quir::symbolizeEnum<quir::TimeUnits>(unit))
        return unitEnum.getValue();
    return failure();
  }
};


} // namespace mlir


#endif // QUIR_QUIRENUMS_H

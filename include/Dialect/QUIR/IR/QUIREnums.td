//===- QUIREnums.td - QUIR dialect enums -------------------*- tablegen -*-===//
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


#ifndef QUIR_ENUMS
#define QUIR_ENUMS

include "Dialect/QUIR/IR/QUIRDialect.td"

include "mlir/IR/EnumAttr.td"

def DurationUnit : I32EnumAttr<
    "DurationUnitEnum", "duration units enum",
    [
      I32EnumAttrCase<"dt", 0>,
      I32EnumAttrCase<"s", 1>,
      I32EnumAttrCase<"ms", 2>,
      I32EnumAttrCase<"us", 3>,
      I32EnumAttrCase<"ns", 4>,
      I32EnumAttrCase<"ps", 5>,
      I32EnumAttrCase<"fs", 6>,

    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::quir";
}

def QUIR_DurationUnitAttr : EnumAttr<
  QUIRDialect, DurationUnit, "duration_unit">;

#endif // QUIR_ENUMS

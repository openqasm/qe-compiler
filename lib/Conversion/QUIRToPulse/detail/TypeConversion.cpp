//===- PlayProcessing.cpp - Pulse dialect ------------------------*- C++-*-===//
//
// (C) Copyright IBM 2023.
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

#include "Conversion/QUIRToPulse/QUIRToPulse.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Optional.h"

namespace mlir::pulse {

static ::llvm::Optional<Type> convertQubitTypes(Type type) {
  if (auto qubitType = type.dyn_cast<quir::QubitType>())
    return PortGroupType::get(type.getContext());
  return llvm::None;
}

static ::llvm::Optional<Type> convertLegalTypes(Type type) {
  if (type.dyn_cast<quir::AngleType>() || type.dyn_cast<IndexType>() ||
      type.dyn_cast<MemRefType>())
    return type;
  return llvm::None;
}

QUIRTypeConverter::QUIRTypeConverter() {
  addConversion(convertQubitTypes);
  addConversion(convertLegalTypes);
}

} // namespace mlir::pulse

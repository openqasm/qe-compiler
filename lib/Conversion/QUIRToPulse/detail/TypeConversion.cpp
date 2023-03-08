//===- PlayProcessing.cpp - Pulse dialect ------------------------*- C++-*-===//
//
// (C) Copyright IBM 2023.
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

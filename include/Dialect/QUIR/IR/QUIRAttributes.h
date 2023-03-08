//===- QUIRAttributes.h - QUIR dialect attributes ---------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
///
///  Attributes for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUIRATTRIBUTES_H
#define QUIR_QUIRATTRIBUTES_H

#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/QUIR/IR/QUIRAttributes.h.inc"

namespace mlir::quir {
static inline llvm::StringRef getInputParameterAttrName() {
  return "quir.inputParameter";
}
static inline llvm::StringRef getPhysicalIdAttrName() {
  return "quir.physicalId";
}
static inline llvm::StringRef getPhysicalIdsAttrName() {
  return "quir.physicalIds";
}
static inline llvm::StringRef getNoReportRuntimeAttrName() {
  return "quir.noReportRuntime";
}
static inline llvm::StringRef getNoReportUserResultAttrName() {
  return "quir.noReportUserResult";
}
} // namespace mlir::quir

#endif // QUIR_QUIRATTRIBUTES_H

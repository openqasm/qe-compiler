//===- QUIRDialect.cpp - QUIR dialect ---------------------------*- C++ -*-===//
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
///  This file defines the QUIR dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRDialect.h"

// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIRAttributes.h"
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIREnums.h"
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIROps.h"
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include <optional>
#include <type_traits>

/// Tablegen Definitions
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIRTypes.cpp.inc"

namespace mlir {
// TODO: This is a parser template for APFloat, not defined in LLVM 14,
// perhaps in future versions? Only able to make this work with `double`
// so anything that requires more precision will need an update or definition
// for parseFloat() that takes APFloat or gnu mpfr.
template <class FloatT>
struct FieldParser<FloatT,
                   std::enable_if_t<std::is_same_v<FloatT, APFloat>, FloatT>> {
  static FailureOr<FloatT> parse(AsmParser &parser) {
    double value;
    if (parser.parseFloat(value))
      return failure();
    return APFloat(value);
  }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// Table generated attribute method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIRAttributes.cpp.inc"

namespace mlir::quir {

//===----------------------------------------------------------------------===//
// Quir dialect.
//===----------------------------------------------------------------------===//

struct QuirInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // inlining for call operations
  auto isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const -> bool final {
    return true;
  }

  /// For now all operations within quir can be inlined.
  auto isLegalToInline(Operation *, Region *, bool, IRMapping &) const
      -> bool final {
    return true;
  }
};

void quir::QUIRDialect::initialize() {

  addTypes<
#define GET_TYPEDEF_LIST
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIRTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIR.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIRAttributes.cpp.inc"
      >();

  addInterfaces<QuirInlinerInterface>();
}

template <typename QUIRType>
mlir::Type parseOptionalWidth(mlir::AsmParser &parser) {
  unsigned width = -1;

  // non-parameterized Qubit type?
  if (parser.parseOptionalLess())
    return QUIRType::get(parser.getContext(), std::nullopt);

  // incorrect syntax
  if (parser.parseInteger(width) || parser.parseGreater())
    return {};

  if (width < 1) {
    parser.emitError(parser.getNameLoc(), "width must be > 0");
    return {};
  }

  return QUIRType::get(parser.getContext(), width);
}

LogicalResult QubitType::verify(function_ref<InFlightDiagnostic()> emitError,
                                int width) {
  if (width <= 0)
    return emitError() << "width must be > 0";
  return success();
}

LogicalResult AngleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                std::optional<int> width) {
  if (width.has_value() && width.value() <= 0)
    return emitError() << "width must be > 0";
  return success();
}

} // namespace mlir::quir

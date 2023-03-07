//===- QUIRDialect.cpp - QUIR dialect ---------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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
///  This file defines the QUIR dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "llvm/ADT/TypeSwitch.h"

/// Tablegen Definitions
#include "Dialect/QUIR/IR/QUIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/QUIR/IR/QUIRTypes.cpp.inc"

namespace mlir {
// TODO: This is a parser template for APFloat, not defined in LLVM 14,
// perhaps in future versions? Only able to make this work with `double`
// so anything that requires more precision will need an update or definition
// for parseFloat() that takes APFloat or gnu mpfr.
template <class FloatT>
struct FieldParser<
    FloatT, std::enable_if_t<std::is_same<FloatT, APFloat>::value, FloatT>> {
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
  auto isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const -> bool final {
    return true;
  }
};

void quir::QUIRDialect::initialize() {

  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/QUIR/IR/QUIRTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/QUIR/IR/QUIR.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/QUIR/IR/QUIRAttributes.cpp.inc"
      >();

  addInterfaces<QuirInlinerInterface>();
}

template <typename QUIRType>
mlir::Type parseOptionalWidth(mlir::AsmParser &parser) {
  unsigned width = -1;

  // non-parameterized Qubit type?
  if (parser.parseOptionalLess())
    return QUIRType::get(parser.getContext(), llvm::None);

  // incorrect syntax
  if (parser.parseInteger(width) || parser.parseGreater())
    return {};

  if (width < 1) {
    parser.emitError(parser.getNameLoc(), "width must be > 0");
    return {};
  }

  return QUIRType::get(parser.getContext(), width);
}

mlir::Type AngleType::parse(mlir::AsmParser &parser) {
  return parseOptionalWidth<AngleType>(parser);
}

static void printOptionalWidth(llvm::Optional<int> width,
                               mlir::AsmPrinter &printer) {
  if (width.hasValue()) {
    printer << "<";
    printer.printStrippedAttrOrType(width);
    printer << ">";
  }
}

void AngleType::print(mlir::AsmPrinter &printer) const {
  printOptionalWidth(getImpl()->width, printer);
}

LogicalResult QubitType::verify(function_ref<InFlightDiagnostic()> emitError,
                                int width) {
  if (width <= 0)
    return emitError() << "width must be > 0";
  return success();
}

LogicalResult AngleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                llvm::Optional<int> width) {
  if (width.hasValue() && width.getValue() <= 0)
    return emitError() << "width must be > 0";
  return success();
}

/// Materialize a constant, can be any buildable type, used by canonicalization
Operation *QUIRDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<quir::ConstantOp>(loc, value, type);
}

} // namespace mlir::quir

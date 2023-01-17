//===- SystemDialect.cpp - System dialect ---------------------------*-
// C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "Dialect/System/IR/SystemDialect.h"
#include "Dialect/System/IR/SystemAttributes.h"
#include "Dialect/System/IR/SystemOps.h"
#include "Dialect/System/IR/SystemTypes.h"

#include "llvm/ADT/TypeSwitch.h"

/// Tablegen Definitions
#include "Dialect/System/IR/SystemDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/System/IR/SystemTypes.cpp.inc"

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
#include "Dialect/System/IR/SystemAttributes.cpp.inc"

namespace mlir::sys {

//===----------------------------------------------------------------------===//
// System dialect.
//===----------------------------------------------------------------------===//

void sys::SystemDialect::initialize() {

  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/System/IR/SystemTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/System/IR/System.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/System/IR/SystemAttributes.cpp.inc"
      >();
}

// /// Materialize a constant, can be any buildable type, used by
// canonicalization Operation *SystemDialect::materializeConstant(OpBuilder
// &builder,
//                                                  Attribute value, Type type,
//                                                  Location loc) {
//   return builder.create<sys::ConstantOp>(loc, value, type);
// }

} // namespace mlir::sys

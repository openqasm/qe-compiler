//===- OQ3Ops.cpp - OpenQASM 3 dialect ops ----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file defines the operations in the OpenQASM 3 dialect.
///
//===----------------------------------------------------------------------===//

#include "Dialect/OQ3/IR/OQ3Ops.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::oq3;

namespace {
mlir::LogicalResult
verifyOQ3VariableOpSymbolUses(SymbolTableCollection &symbolTable,
                              mlir::Operation *op,
                              bool operandMustMatchSymbolType = false) {
  assert(op);

  // Check that op has attribute variable_name
  auto varRefAttr = op->getAttrOfType<FlatSymbolRefAttr>("variable_name");
  if (!varRefAttr)
    return op->emitOpError(
        "requires a symbol reference attribute 'variable_name'");

  // Check that symbol reference resolves to a variable declaration
  auto declOp =
      symbolTable.lookupNearestSymbolFrom<DeclareVariableOp>(op, varRefAttr);
  if (!declOp)
    return op->emitOpError() << "no valid reference to a variable '"
                             << varRefAttr.getValue() << "'";

  assert(op->getNumResults() <= 1 && "assume none or single result");

  // Check that type of variables matches result type of this Op
  if (op->getNumResults() == 1) {
    if (op->getResult(0).getType() != declOp.getType())
      return op->emitOpError(
          "type mismatch between variable declaration and variable use");
  }

  if (op->getNumOperands() > 0 && operandMustMatchSymbolType) {
    assert(op->getNumOperands() == 1 &&
           "type check only supported for a single operand");
    if (op->getOperand(0).getType() != declOp.getType())
      return op->emitOpError(
          "type mismatch between variable declaration and variable assignment");
  }

  // tbd also check types for assigning ops (once we have an interface
  // OQ3VariableOps with bool predicates for assigning / referencing ops)

  return success();
}
} // anonymous namespace

//===----------------------------------------------------------------------===//
// CBit ops
//===----------------------------------------------------------------------===//

namespace {
std::optional<mlir::Value> findDefiningBitInBitmap(mlir::Value val,
                                                   mlir::IntegerAttr bitIndex) {

  // for single-bit registers, CBitExtractBitOp is the identity.
  if (val.getType().isInteger(1))
    return val;

  mlir::Operation *op = val.getDefiningOp();

  // follow chains of CBit_InsertBit operations and try to find one matching the
  // requested bit
  while (auto insertBitOp = mlir::dyn_cast_or_null<CBitInsertBitOp>(op)) {
    if (insertBitOp.getIndexAttr() == bitIndex)
      return insertBitOp.getAssignedBit();

    op = insertBitOp.getOperand().getDefiningOp();
  }

  // did we identify an op that provides the single bit?
  if (op && op->getResult(0).getType().isInteger(1))
    return op->getResult(0);

  return std::nullopt;
}
} // anonymous namespace

::mlir::OpFoldResult CBitExtractBitOp::fold(FoldAdaptor adaptor) {

  auto foundDefiningBitOrNone =
      findDefiningBitInBitmap(getOperand(), getIndexAttr());

  if (foundDefiningBitOrNone)
    return foundDefiningBitOrNone.value();
  return nullptr;
}

mlir::LogicalResult CBitExtractBitOp::verify() {

  auto t = getOperand().getType();

  if (auto cbitType = t.dyn_cast<mlir::quir::CBitType>();
      cbitType && getIndex().ult(cbitType.getWidth()))
    return success();

  if (t.isIntOrIndex() && getIndex().ult(t.getIntOrFloatBitWidth()))
    return success();

  return emitOpError("index must be less than the width of the operand.");
}

mlir::LogicalResult
CBitAssignBitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

//===----------------------------------------------------------------------===//
// Variable ops
//===----------------------------------------------------------------------===//

mlir::LogicalResult DeclareVariableOp::verify() {
  auto t = (*this).getType();

  if (t.isa<::mlir::quir::AngleType>() || t.isa<::mlir::quir::CBitType>() ||
      t.isa<::mlir::quir::DurationType>() ||
      t.isa<::mlir::quir::StretchType>() || t.isIntOrIndexOrFloat() ||
      t.isa<ComplexType>())
    return success();
  std::string str;
  llvm::raw_string_ostream os(str);
  t.print(os);

  return emitOpError("MLIR type " + str + " not supported for declarations.");
}

mlir::LogicalResult
VariableAssignOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation(), true);
}

mlir::LogicalResult
VariableLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

//===----------------------------------------------------------------------===//
// Array ops
//===----------------------------------------------------------------------===//

mlir::LogicalResult DeclareArrayOp::verify() {
  auto t = (*this).getType();

  if (t.isa<::mlir::quir::AngleType>() || t.isa<::mlir::quir::CBitType>() ||
      t.isa<::mlir::quir::DurationType>() ||
      t.isa<::mlir::quir::StretchType>() || t.isIntOrIndexOrFloat())
    return success();
  return failure();
}

mlir::LogicalResult
AssignArrayElementOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

mlir::LogicalResult
UseArrayElementOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

//===----------------------------------------------------------------------===//
// Binary / Unary Ops
//===----------------------------------------------------------------------===//

mlir::LogicalResult AngleCmpOp::verify() {
  std::vector predicates = {"eq",  "ne",  "slt", "sle", "sgt",
                            "sge", "ult", "ule", "ugt", "uge"};

  if (std::find(predicates.begin(), predicates.end(), getPredicate()) !=
      predicates.end())
    return success();

  return emitOpError("requires predicate \"eq\", \"ne\", \"slt\", \"sle\", "
                     "\"sgt\", \"sge\", \"ult\", \"ule\", \"ugt\", \"uge\"");
}

#define GET_OP_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/OQ3/IR/OQ3Ops.cpp.inc"

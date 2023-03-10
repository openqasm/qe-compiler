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
#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/OQ3/IR/OQ3Types.h"

using namespace mlir;
using namespace mlir::oq3;

static LogicalResult
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
    if (op->getResult(0).getType() != declOp.type())
      return op->emitOpError(
          "type mismatch between variable declaration and variable use");
  }

  if (op->getNumOperands() > 0 && operandMustMatchSymbolType) {
    assert(op->getNumOperands() == 1 &&
           "type check only supported for a single operand");
    if (op->getOperand(0).getType() != declOp.type())
      return op->emitOpError(
          "type mismatch between variable declaration and variable assignment");
  }

  // tbd also check types for assigning ops (once we have an interface
  // OQ3VariableOps with bool predicates for assigning / referencing ops)

  return success();
}

//===----------------------------------------------------------------------===//
// CBit ops
//===----------------------------------------------------------------------===//

static llvm::Optional<mlir::Value>
findDefiningBitInBitmap(mlir::Value val, mlir::IntegerAttr bitIndex) {

  mlir::Operation *op = val.getDefiningOp();

  // follow chains of CBit_InsertBit operations and try to find one matching the
  // requested bit
  while (auto insertBitOp = mlir::dyn_cast_or_null<CBitInsertBitOp>(op)) {
    if (insertBitOp.indexAttr() == bitIndex)
      return insertBitOp.assigned_bit();

    op = insertBitOp.operand().getDefiningOp();
  }

  // is the value defined by an i1 constant? then that would be the bit
  if (auto constantOp =
          mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(op)) {
    if (constantOp.getType().isInteger(1))
      return constantOp.getResult();
  }

  return llvm::None;
}

::mlir::OpFoldResult
CBitExtractBitOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {

  auto foundDefiningBitOrNone = findDefiningBitInBitmap(operand(), indexAttr());

  if (foundDefiningBitOrNone)
    return foundDefiningBitOrNone.getValue();
  return nullptr;
}

LogicalResult
CBitAssignBitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

//===----------------------------------------------------------------------===//
// Variable ops
//===----------------------------------------------------------------------===//

LogicalResult
VariableAssignOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation(), true);
}

LogicalResult
UseVariableOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

//===----------------------------------------------------------------------===//
// Array ops
//===----------------------------------------------------------------------===//

LogicalResult
AssignArrayElementOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

LogicalResult
UseArrayElementOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyOQ3VariableOpSymbolUses(symbolTable, getOperation());
}

#define GET_OP_CLASSES
#include "Dialect/OQ3/IR/OQ3Ops.cpp.inc"

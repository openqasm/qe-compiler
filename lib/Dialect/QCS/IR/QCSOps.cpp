//===- QCSOps.cpp - Quantum Control System dialect ops ----------*- C++ -*-===//
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
/// This file defines the operations in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QCS/IR/QCSTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

using namespace mlir;
using namespace mlir::qcs;

#define GET_OP_CLASSES
#include "Dialect/QCS/IR/QCSOps.cpp.inc"

static LogicalResult
verifyQCSParameterOpSymbolUses(SymbolTableCollection &symbolTable,
                              mlir::Operation *op,
                              bool operandMustMatchSymbolType = false) {
  assert(op);

  // Check that op has attribute variable_name
  auto paramRefAttr = op->getAttrOfType<FlatSymbolRefAttr>("parameter_name");
  if (!paramRefAttr)
    return op->emitOpError(
        "requires a symbol reference attribute 'parameter_name'");

  // Check that symbol reference resolves to a parameter declaration
  auto declOp =
      symbolTable.lookupNearestSymbolFrom<InputParameterOp>(op, paramRefAttr);
  if (!declOp)
    return op->emitOpError() << "no valid reference to a parameter '"
                             << paramRefAttr.getValue() << "'";

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
  return success();
}

LogicalResult
ParameterLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyQCSParameterOpSymbolUses(symbolTable, getOperation(), true);
}

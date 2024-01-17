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

#include "Dialect/QCS/IR/QCSTypes.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LogicalResult.h>

#include "llvm/ADT/StringMap.h"

#include <cassert>

using namespace mlir;
using namespace mlir::qcs;

#define GET_OP_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSOps.cpp.inc"

namespace {
LogicalResult
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
      symbolTable.lookupNearestSymbolFrom<DeclareParameterOp>(op, paramRefAttr);

  // check higher level modules
  if (!declOp) {
    auto targetModuleOp = op->getParentOfType<mlir::ModuleOp>();
    if (targetModuleOp) {
      auto topLevelModuleOp = targetModuleOp->getParentOfType<mlir::ModuleOp>();
      if (!declOp && topLevelModuleOp)
        declOp = symbolTable.lookupNearestSymbolFrom<DeclareParameterOp>(
            topLevelModuleOp, paramRefAttr);
    }
  }

  if (!declOp)
    return op->emitOpError() << "no valid reference to a parameter '"
                             << paramRefAttr.getValue() << "'";

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
  return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// ParameterLoadOp
//===----------------------------------------------------------------------===//

LogicalResult
ParameterLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyQCSParameterOpSymbolUses(symbolTable, getOperation(), true);
}

// Returns the float value from the initial value of this parameter
ParameterType ParameterLoadOp::getInitialValue() {
  auto *op = getOperation();
  auto paramRefAttr =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("parameter_name");
  auto declOp =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::qcs::DeclareParameterOp>(
          op, paramRefAttr);

  // check higher level modules

  auto currentScopeOp = op->getParentOfType<mlir::ModuleOp>();
  do {
    declOp = mlir::SymbolTable::lookupNearestSymbolFrom<
        mlir::qcs::DeclareParameterOp>(currentScopeOp, paramRefAttr);
    if (declOp)
      break;
    currentScopeOp = currentScopeOp->getParentOfType<mlir::ModuleOp>();
    assert(currentScopeOp);
  } while (!declOp);

  assert(declOp);

  double retVal;

  auto iniValue = declOp.getInitialValue();
  if (iniValue.has_value()) {
    auto angleAttr = iniValue.value().dyn_cast<mlir::quir::AngleAttr>();

    auto floatAttr = iniValue.value().dyn_cast<FloatAttr>();

    if (!(angleAttr || floatAttr)) {
      op->emitError(
          "Parameters are currently limited to angles or float[64] only.");
      return 0.0;
    }

    if (angleAttr)
      retVal = angleAttr.getValue().convertToDouble();

    if (floatAttr)
      retVal = floatAttr.getValue().convertToDouble();

    return retVal;
  }

  op->emitError("Does not have initial value set.");
  return 0.0;
}

// Returns the float value from the initial value of this parameter
// this version uses a precomputed map of parameter_name to the initial_value
// in order to avoid slow SymbolTable lookups
ParameterType ParameterLoadOp::getInitialValue(
    llvm::StringMap<ParameterType> &declareParametersMap) {
  auto *op = getOperation();
  auto paramRefAttr =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("parameter_name");

  auto paramOpEntry = declareParametersMap.find(paramRefAttr.getValue());

  if (paramOpEntry == declareParametersMap.end()) {
    op->emitError("Could not find declare parameter op " +
                  paramRefAttr.getValue().str());
    return 0.0;
  }

  return paramOpEntry->second;
}

//===----------------------------------------------------------------------===//
// End ParameterLoadOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ParallelEndOp
//===----------------------------------------------------------------------===//

mlir::ParseResult ParallelEndOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return mlir::success();
}

void ParallelEndOp::print(mlir::OpAsmPrinter &printer) {
  printer << getOperationName();
}

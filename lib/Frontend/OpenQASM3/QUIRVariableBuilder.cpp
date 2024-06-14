//===- QUIRVariableBuilder.cpp ----------------------------------*- C++ -*-===//
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
///  This file contains the implementation of the generator for variable
///  handling code QUIRVariableBuilder.
///
//===----------------------------------------------------------------------===//

#include "Frontend/OpenQASM3/QUIRVariableBuilder.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <qasm/AST/ASTResult.h>
#include <qasm/AST/ASTSymbolTable.h>
#include <qasm/AST/ASTTypeEnums.h>
#include <qasm/AST/ASTTypes.h>

namespace qssc::frontend::openqasm3 {

void QUIRVariableBuilder::generateVariableDeclaration(
    mlir::Location location, llvm::StringRef variableName, mlir::Type type,
    bool isInputVariable, bool isOutputVariable) {

  // Input variables are not used as parameter loads replace them
  // for performance reasons.
  // TODO: Replace many parameters with array accesses.
  if (isInputVariable)
    return;
  // variables are symbols and thus need to be placed directly in a surrounding
  // Op that contains a symbol table.
  mlir::OpBuilder::InsertionGuard const g(builder);
  auto *symbolTableOp = mlir::SymbolTable::getNearestSymbolTable(
      builder.getInsertionBlock()->getParentOp());
  assert(symbolTableOp &&
         "require surrounding op with a symbol table (should be the Module)");
  auto surroundingModuleOp = mlir::dyn_cast<mlir::ModuleOp>(*symbolTableOp);
  assert(surroundingModuleOp && "assume symbol table residing in module");
  builder.setInsertionPoint(&surroundingModuleOp.front());

  auto declareOp = builder.create<mlir::oq3::DeclareVariableOp>(
      location, variableName, mlir::TypeAttr::get(type));

  if (lastDeclaration.count(surroundingModuleOp))
    declareOp->moveAfter(lastDeclaration[surroundingModuleOp]);

  lastDeclaration[surroundingModuleOp] = declareOp; // save this to insert after

  if (isOutputVariable)
    declareOp.setOutputAttr(builder.getUnitAttr());
  variables.emplace(variableName.str(), type);
}

void QUIRVariableBuilder::generateParameterDeclaration(
    mlir::Location location, llvm::StringRef variableName, mlir::Type type,
    mlir::Value assignedValue) {

  mlir::OpBuilder::InsertionGuard const g(builder);
  auto *symbolTableOp = mlir::SymbolTable::getNearestSymbolTable(
      builder.getInsertionBlock()->getParentOp());
  assert(symbolTableOp &&
         "require surrounding op with a symbol table (should be the Module)");
  auto surroundingModuleOp = mlir::dyn_cast<mlir::ModuleOp>(*symbolTableOp);
  assert(surroundingModuleOp && "assume symbol table residing in module");
  builder.setInsertionPoint(&surroundingModuleOp.front());

  mlir::qcs::DeclareParameterOp declareParameterOp;

  // add qcs input parameter
  if (auto constantOp = mlir::dyn_cast<mlir::quir::ConstantOp>(
          assignedValue.getDefiningOp())) {
    declareParameterOp =
        getClassicalBuilder().create<mlir::qcs::DeclareParameterOp>(
            location, variableName.str(),
            builder.getType<mlir::quir::AngleType>(64), constantOp.getValue());
  }

  // if the source is a arith::ConstantOp cast to angle
  if (auto constantOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
          assignedValue.getDefiningOp())) {
    declareParameterOp =
        getClassicalBuilder().create<mlir::qcs::DeclareParameterOp>(
            location, variableName.str(), constantOp.getType(),
            constantOp.getValue());
  }

  declareParameterOp->moveBefore(lastDeclaration[surroundingModuleOp]);
  lastDeclaration[surroundingModuleOp] = declareParameterOp;
}

mlir::Value
QUIRVariableBuilder::generateParameterLoad(mlir::Location location,
                                           llvm::StringRef variableName) {

  auto op = getClassicalBuilder().create<mlir::qcs::ParameterLoadOp>(
      location, builder.getType<mlir::quir::AngleType>(64), variableName.str());

  return op;
}

void QUIRVariableBuilder::generateArrayVariableDeclaration(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Type elementType, int64_t width) {

  builder.create<mlir::oq3::DeclareArrayOp>(
      location, builder.getStringAttr(variableName),
      mlir::TypeAttr::get(elementType), builder.getIndexAttr(width));
  variables.emplace(
      variableName.str(),
      mlir::RankedTensorType::get(mlir::ArrayRef<int64_t>{width}, elementType));
}

void QUIRVariableBuilder::generateVariableAssignment(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Value assignedValue) {

  getClassicalBuilder().create<mlir::oq3::VariableAssignOp>(
      location, variableName, assignedValue);
}

void QUIRVariableBuilder::generateArrayVariableElementAssignment(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Value assignedValue, size_t elementIndex) {

  builder.create<mlir::oq3::AssignArrayElementOp>(
      location,
      mlir::FlatSymbolRefAttr::get(builder.getStringAttr(variableName)),
      builder.getIndexAttr(elementIndex), assignedValue);
}

void QUIRVariableBuilder::generateCBitSingleBitAssignment(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Value assignedValue, size_t bitPosition, size_t registerWidth) {

  // TODO at some point, implement any follow-up changes required and move away
  // from CBitAssignBitOp.
#if 0
  auto oldCBitValue = generateVariableUse(location, variableName, builder.getType<mlir::quir::CBitType>(registerWidth));
  auto cbitWithInsertedBit = builder.create<mlir::oq3::CBitInsertBitOp>(
            location, oldCBitValue.getType(), oldCBitValue,
            assignedValue, builder.getIndexAttr(bitPosition));

  builder.create<mlir::oq3::VariableAssignOp>(
        location, mlir::SymbolRefAttr::get(builder.getStringAttr(variableName)), cbitWithInsertedBit);

#else
  getClassicalBuilder().create<mlir::oq3::CBitAssignBitOp>(
      location,
      mlir::SymbolRefAttr::get(
          getClassicalBuilder().getStringAttr(variableName)),
      getClassicalBuilder().getIndexAttr(bitPosition),
      getClassicalBuilder().getIndexAttr(registerWidth), assignedValue);
#endif
}

mlir::Value
QUIRVariableBuilder::generateVariableUse(mlir::Location location,
                                         llvm::StringRef variableName,
                                         mlir::Type variableType) {
  return getClassicalBuilder().create<mlir::oq3::VariableLoadOp>(
      location, variableType, variableName);
}

mlir::Value QUIRVariableBuilder::generateArrayVariableElementUse(
    mlir::Location location, llvm::StringRef variableName, size_t elementIndex,
    mlir::Type elementType) {

  return builder.create<mlir::oq3::UseArrayElementOp>(
      location, elementType,
      mlir::SymbolRefAttr::get(builder.getStringAttr(variableName)),
      builder.getIndexAttr(elementIndex));
}

mlir::Type
QUIRVariableBuilder::resolveQUIRVariableType(QASM::ASTType astType,
                                             const unsigned bits) const {
  switch (astType) {
  case QASM::ASTTypeBool:
    return builder.getI1Type();

  case QASM::ASTTypeBitset: {
    return builder.getType<mlir::quir::CBitType>(bits);
  }

  case QASM::ASTTypeMPInteger:
  case QASM::ASTTypeInt: {
    return builder.getIntegerType(bits);
  }

  case QASM::ASTTypeFloat:
  case QASM::ASTTypeMPDecimal:
  case QASM::ASTTypeMPComplex: {
    mlir::FloatType floatType;
    if (bits <= 16)
      floatType = builder.getF16Type();
    else if (bits <= 32)
      floatType = builder.getF32Type();
    else if (bits <= 64)
      floatType = builder.getF64Type();
    else if (bits <= 80)
      floatType = builder.getF80Type();
    else if (bits <= 128)
      floatType = builder.getF128Type();
    else
      return builder.getNoneType();
    if (astType == QASM::ASTTypeMPComplex)
      return mlir::ComplexType::get(floatType);
    return floatType;
  }

  case QASM::ASTTypeAngle: {
    return builder.getType<mlir::quir::AngleType>(bits);
  }

  default:
    return builder.getNoneType();
  }
}

mlir::Type QUIRVariableBuilder::resolveQUIRVariableType(
    QASM::ASTSymbolTableEntry const *entry) const {
  assert(entry && "symbol table entry must not be null.");
  assert(entry->GetIdentifier() &&
         "Symbol table entry must have an assigned identifier");
  return resolveQUIRVariableType(entry->GetValueType(),
                                 entry->GetIdentifier()->GetBits());
}

mlir::Type QUIRVariableBuilder::resolveQUIRVariableType(
    const QASM::ASTDeclarationNode *node) {
  assert(node && "node argument must be non-null");
  assert(node->GetIdentifier() &&
         "node argument must have an assigned identifier");
  return resolveQUIRVariableType(node->GetASTType(),
                                 node->GetIdentifier()->GetBits());
}

mlir::Type
QUIRVariableBuilder::resolveQUIRVariableType(const QASM::ASTResultNode *node) {
  assert(node && "node argument must be non-null");
  return resolveQUIRVariableType(node->GetResultType(), node->GetResultBits());
}

} // namespace qssc::frontend::openqasm3

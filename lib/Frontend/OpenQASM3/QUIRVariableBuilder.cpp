//===- QUIRVariableBuilder.cpp ----------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
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

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "qasm/AST/ASTTypeEnums.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <string>
#include <utility>

namespace qssc::frontend::openqasm3 {

void QUIRVariableBuilder::generateVariableDeclaration(
    mlir::Location location, llvm::StringRef variableName, mlir::Type type,
    bool isInputVariable, bool isOutputVariable) {

  // variables are symbols and thus need to be placed directly in a surrounding
  // Op that contains a symbol table.
  mlir::OpBuilder::InsertionGuard g(builder);
  auto *symbolTableOp = mlir::SymbolTable::getNearestSymbolTable(
      builder.getInsertionBlock()->getParentOp());
  assert(symbolTableOp &&
         "require surrounding op with a symbol table (should be the Module)");
  auto surroundingModuleOp = mlir::dyn_cast<mlir::ModuleOp>(*symbolTableOp);
  assert(surroundingModuleOp && "assume symbol table residing in module");
  builder.setInsertionPoint(&surroundingModuleOp.front());

  auto declareOp = builder.create<mlir::quir::DeclareVariableOp>(
      location, variableName, mlir::TypeAttr::get(type));

  if (lastDeclaration.count(surroundingModuleOp))
    declareOp->moveAfter(lastDeclaration[surroundingModuleOp]);

  lastDeclaration[surroundingModuleOp] = declareOp; // save this to insert after

  if (isInputVariable)
    declareOp.inputAttr(builder.getUnitAttr());
  if (isOutputVariable)
    declareOp.outputAttr(builder.getUnitAttr());
  variables.emplace(variableName.str(), type);
}

void QUIRVariableBuilder::generateArrayVariableDeclaration(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Type elementType, int64_t width) {

  builder.create<mlir::quir::DeclareArrayOp>(
      location, builder.getStringAttr(variableName),
      mlir::TypeAttr::get(elementType), builder.getIndexAttr(width));
  variables.emplace(
      variableName.str(),
      mlir::RankedTensorType::get(mlir::ArrayRef<int64_t>{width}, elementType));
}

void QUIRVariableBuilder::generateVariableAssignment(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Value assignedValue) {

  builder.create<mlir::quir::VariableAssignOp>(location, variableName,
                                               assignedValue);
}

void QUIRVariableBuilder::generateArrayVariableElementAssignment(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Value assignedValue, size_t elementIndex) {

  builder.create<mlir::quir::AssignArrayElementOp>(
      location,
      mlir::FlatSymbolRefAttr::get(builder.getStringAttr(variableName)),
      builder.getIndexAttr(elementIndex), assignedValue);
}

void QUIRVariableBuilder::generateCBitSingleBitAssignment(
    mlir::Location location, llvm::StringRef variableName,
    mlir::Value assignedValue, size_t bitPosition, size_t registerWidth) {

  // TODO at some point, implement any follow-up changes required and move away
  // from AssignCbitBitOp.
#if 0
  auto oldCbitValue = generateVariableUse(location, variableName, builder.getType<mlir::quir::CBitType>(registerWidth));
  auto cbitWithInsertedBit = builder.create<mlir::quir::CBit_InsertBitOp>(
            location, oldCbitValue.getType(), oldCbitValue,
            assignedValue, builder.getIndexAttr(bitPosition));

  builder.create<mlir::quir::VariableAssignOp>(
        location, mlir::SymbolRefAttr::get(builder.getStringAttr(variableName)), cbitWithInsertedBit);

#else
  builder.create<mlir::quir::AssignCbitBitOp>(
      location, mlir::SymbolRefAttr::get(builder.getStringAttr(variableName)),
      builder.getIndexAttr(bitPosition), builder.getIndexAttr(registerWidth),
      assignedValue);
#endif
}

mlir::Value
QUIRVariableBuilder::generateVariableUse(mlir::Location location,
                                         llvm::StringRef variableName,
                                         mlir::Type variableType) {
  return builder.create<mlir::quir::UseVariableOp>(location, variableType,
                                                   variableName);
}

mlir::Value QUIRVariableBuilder::generateArrayVariableElementUse(
    mlir::Location location, llvm::StringRef variableName, size_t elementIndex,
    mlir::Type elementType) {

  return builder.create<mlir::quir::UseArrayElementOp>(
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

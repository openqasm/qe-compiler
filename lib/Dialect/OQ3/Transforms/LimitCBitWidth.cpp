//===- LimitCBitWidth.cpp - limit width of cbit registers -------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the pass limiting the width of classical bit
///  registers. Registers that are larger than the limit are broken into
///  multiple registers of the maximum size.
///
//===----------------------------------------------------------------------===//

#include "Dialect/OQ3/Transforms/LimitCBitWidth.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"

#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "LimitCBitWidth"

using namespace mlir::oq3;

void LimitCBitWidthPass::addNewDeclareVariableOps(
    Operation *module, DeclareVariableOp op, uint numRegistersRequired,
    uint numRemainingBits,
    llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters,
    mlir::SymbolTableCollection &symbolTableCol) {
  auto variableName = op.getSymName();

  // create new registers with _# added to end
  // add additional "_" if symbol name is found
  OpBuilder builder(op);
  newRegisters.clear();
  for (uint regNum = 0; regNum < numRegistersRequired; regNum++) {
    std::string newVariableName =
        variableName.str() + "_" + std::to_string(regNum);
    auto *findOp =
        symbolTableCol.getSymbolTable(module).lookup(newVariableName);
    if (findOp) {
      uint extraInt = 0;
      std::string const baseVariableName = newVariableName;
      while (findOp) {
        newVariableName = baseVariableName + std::to_string(extraInt++);
        findOp = symbolTableCol.getSymbolTable(module).lookup(newVariableName);
      }
    }

    uint const bitWidth =
        getNewRegisterWidth(regNum, numRegistersRequired, numRemainingBits);
    auto newCbitType = builder.getType<mlir::quir::CBitType>(bitWidth);
    newRegisters.push_back(builder.create<DeclareVariableOp>(
        op->getLoc(), newVariableName, mlir::TypeAttr::get(newCbitType)));
  }
}

void LimitCBitWidthPass::processOp(
    CBitAssignBitOp cbitAssignOp, uint numRegistersRequired,
    uint numRemainingBits,
    llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters) {
  uint64_t index;
  uint64_t reg;
  std::tie(reg, index) = remapBit(cbitAssignOp.getIndex());
  auto width =
      newRegisters[reg].getType().dyn_cast<quir::CBitType>().getWidth();
  auto value = cbitAssignOp.getAssignedBit();

  OpBuilder builder(cbitAssignOp);
  builder.create<CBitAssignBitOp>(
      cbitAssignOp->getLoc(),
      mlir::SymbolRefAttr::get(
          builder.getStringAttr(newRegisters[reg].getSymName())),
      builder.getIndexAttr(index), builder.getIndexAttr(width), value);

  eraseList_.push_back(cbitAssignOp);
}

void LimitCBitWidthPass::processOp(
    VariableAssignOp variableAssignOp, uint orgWidth, uint numRegistersRequired,
    uint numRemainingBits,
    llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters) {
  auto castOp = dyn_cast<oq3::CastOp>(
      variableAssignOp.getAssignedValue().getDefiningOp());
  if (!castOp) {
    variableAssignOp.emitError(
        "expect assigned_value() to be defined by a CastOp");
    signalPassFailure();
  }
  auto constantOp =
      dyn_cast<arith::ConstantOp>(castOp.getArg().getDefiningOp());
  if (!constantOp) {
    castOp.emitError("expect cast arg() to be a constant op");
    signalPassFailure();
  }
  OpBuilder builder(variableAssignOp);
  if (constantOp.getValue().getType() != builder.getIntegerType(orgWidth)) {
    constantOp.emitError("constant type is in valid must have width " +
                         std::to_string(orgWidth));
    signalPassFailure();
  }

  auto intAttr = constantOp.getValue().dyn_cast_or_null<IntegerAttr>();
  if (!intAttr) {
    constantOp.emitError("constant must have a IntegerAttr value");
    signalPassFailure();
  }

  APInt const apInt = intAttr.getValue();

  for (uint regNum = 0; regNum < numRegistersRequired; regNum++) {
    uint const bitWidth =
        getNewRegisterWidth(regNum, numRegistersRequired, numRemainingBits);
    auto subPart = apInt.extractBits(bitWidth, regNum * MAX_CBIT_WIDTH);
    auto initializerVal = builder.create<mlir::arith::ConstantOp>(
        constantOp->getLoc(),
        builder.getIntegerAttr(builder.getIntegerType(bitWidth), subPart));

    auto newCastOp = builder.create<mlir::oq3::CastOp>(
        constantOp->getLoc(), builder.getType<mlir::quir::CBitType>(bitWidth),
        initializerVal);

    builder.create<VariableAssignOp>(variableAssignOp->getLoc(),
                                     newRegisters[regNum].getSymName(),
                                     newCastOp);
  }
  eraseList_.push_back(variableAssignOp);
}

void LimitCBitWidthPass::processOp(
    VariableLoadOp variableLoadOp, uint numRegistersRequired,
    uint numRemainingBits,
    llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters) {
  // load all of the registers
  llvm::SmallVector<VariableLoadOp> newVariableLoads;
  OpBuilder builder(variableLoadOp);
  for (uint regNum = 0; regNum < numRegistersRequired; regNum++) {
    uint const bitWidth =
        getNewRegisterWidth(regNum, numRegistersRequired, numRemainingBits);
    newVariableLoads.push_back(builder.create<VariableLoadOp>(
        variableLoadOp.getLoc(),
        builder.getType<mlir::quir::CBitType>(bitWidth),
        newRegisters[regNum].getSymName()));
  }

  for (auto *loadUse : variableLoadOp->getUsers()) {
    auto extractBitOp = dyn_cast<CBitExtractBitOp>(loadUse);
    if (extractBitOp) {
      uint64_t reg;
      uint64_t remain;
      std::tie(reg, remain) = remapBit(extractBitOp.getIndex());
      auto newExtract = builder.create<CBitExtractBitOp>(
          extractBitOp->getLoc(), builder.getI1Type(), newVariableLoads[reg],
          builder.getIndexAttr(remain));
      extractBitOp->replaceAllUsesWith(newExtract);
      eraseList_.push_back(extractBitOp);
    } else {
      llvm::errs() << "Unhandled use type: ";
      loadUse->dump();
      assert(false);
    }
  }
  eraseList_.push_back(variableLoadOp);
}

std::pair<uint64_t, uint64_t>
LimitCBitWidthPass::remapBit(const llvm::APInt &indexInt) {
  uint64_t index = indexInt.getZExtValue();
  uint64_t const reg = index / MAX_CBIT_WIDTH;
  index = index - (reg * MAX_CBIT_WIDTH);
  return std::make_pair(reg, index);
}

uint LimitCBitWidthPass::getNewRegisterWidth(uint regNum,
                                             uint numRegistersRequired,
                                             uint numRemainingBits) {
  uint const bitWidth =
      regNum == (numRegistersRequired - 1) ? numRemainingBits : MAX_CBIT_WIDTH;
  return bitWidth;
}

void LimitCBitWidthPass::runOnOperation() {

  eraseList_.clear();

  // check for command line override of MAX_CBIT_WIDTH
  if (maxCBitWidthOption.hasValue())
    MAX_CBIT_WIDTH = maxCBitWidthOption.getValue();

  Operation *module = getOperation();

  mlir::SymbolTableCollection symbolTableCol;

  module->walk([&](DeclareVariableOp op) {
    // look for declare variables of CBitType and Width > 64
    auto cbitType = op.getType().dyn_cast<quir::CBitType>();
    if (!cbitType || cbitType.getWidth() <= MAX_CBIT_WIDTH)
      return;

    uint const orgWidth = cbitType.getWidth();
    uint const numRegistersRequired = cbitType.getWidth() / MAX_CBIT_WIDTH + 1;
    uint const numRemainingBits = cbitType.getWidth() % MAX_CBIT_WIDTH;
    llvm::SmallVector<mlir::oq3::DeclareVariableOp> newRegisters;

    addNewDeclareVariableOps(module, op, numRegistersRequired, numRemainingBits,
                             newRegisters, symbolTableCol);

    if (auto rangeOrNone = SymbolTable::getSymbolUses(op, module))
      for (auto &use : rangeOrNone.value()) {
        auto *user = use.getUser();
        if (auto variableAssignOp = dyn_cast<VariableAssignOp>(user))
          processOp(variableAssignOp, orgWidth, numRegistersRequired,
                    numRemainingBits, newRegisters);
        else if (auto cbitAssignOp = dyn_cast<CBitAssignBitOp>(user))
          processOp(cbitAssignOp, numRegistersRequired, numRemainingBits,
                    newRegisters);
        else if (auto variableLoadOp = dyn_cast<VariableLoadOp>(user))
          processOp(variableLoadOp, numRegistersRequired, numRemainingBits,
                    newRegisters);
        else {
          llvm::errs() << "Unhandled use type: ";
          use.getUser()->dump();
          assert(false);
        }
      }
    eraseList_.push_back(op);
  });

  for (auto *op : eraseList_) {
    assert(op->use_empty() && "operation usage expected to be empty");
    op->erase();
  }

} // runOnOperation

llvm::StringRef LimitCBitWidthPass::getArgument() const {
  return "oq3-limit-cbit-width";
}

llvm::StringRef LimitCBitWidthPass::getDescription() const {
  return "Limit classical bit register width";
}

llvm::StringRef LimitCBitWidthPass::getName() const {
  return "Limit CBit Width Pass (" + getArgument().str() + ")";
}

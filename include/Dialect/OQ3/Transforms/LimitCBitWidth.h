//===- LimitCBitWidth.h - limit width of cbit registers --------*- C++ -*--===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the pass limiting the width of classical bit registers.
///  Registers that are larger than the limit are broken into multiple registers
///  of the maximum size.
///
//===----------------------------------------------------------------------===//

#ifndef LIMIT_CBIT_WIDTH_H
#define LIMIT_CBIT_WIDTH_H

#include "Dialect/OQ3/IR/OQ3Ops.h"

#include "mlir/Pass/Pass.h"

namespace mlir::oq3 {

using namespace mlir;

class LimitCBitWidthPass
    : public PassWrapper<LimitCBitWidthPass, OperationPass<ModuleOp>> {
public:
  uint MAX_CBIT_WIDTH{32};

  LimitCBitWidthPass() = default;
  LimitCBitWidthPass(uint maxCBitWidth) : MAX_CBIT_WIDTH(maxCBitWidth) {}
  LimitCBitWidthPass(const LimitCBitWidthPass &pass) : PassWrapper(pass) {}

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName = "Limit CBit Width Pass (" + getArgument().str() + ")";

  Option<uint> maxCBitWidthOption{
      *this, "max-cbit-width",
      llvm::cl::desc("Maximum width of classical bit arrays")};

private:
  void addNewDeclareVariableOps(
      mlir::Operation *module, mlir::oq3::DeclareVariableOp op,
      uint numRegistersRequired, uint numRemainingBits,
      llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters,
      mlir::SymbolTableCollection &symbolTableCol);
  void processOp(mlir::oq3::CBitAssignBitOp cbitAssignOp,
                 uint numRegistersRequired, uint numRemainingBits,
                 llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters);
  void processOp(mlir::oq3::VariableAssignOp variableAssignOp, uint orgWidth,
                 uint numRegistersRequired, uint numRemainingBits,
                 llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters);
  void processOp(mlir::oq3::VariableLoadOp variableLoadOp,
                 uint numRegistersRequired, uint numRemainingBits,
                 llvm::SmallVector<mlir::oq3::DeclareVariableOp> &newRegisters);
  std::pair<uint64_t, uint64_t> remapBit(const llvm::APInt &indexInt);
  uint getNewRegisterWidth(uint regNum, uint numRegistersRequired,
                           uint numRemainingBits);
  std::vector<Operation *> eraseList_;
};
} // namespace mlir::oq3

#endif // LIMIT_CBIT_WIDTH_H

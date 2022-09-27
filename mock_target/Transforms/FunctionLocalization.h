//===- FunctionLocalization.h - Localizing function defs --------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file declares the pass for localizing function defs
//
//===----------------------------------------------------------------------===//

#ifndef MOCK_FUNCTION_LOCALIZATION_H
#define MOCK_FUNCTION_LOCALIZATION_H

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

#include <list>

namespace qssc::targets::mock {

struct SymbolTableBuildPass
    : public mlir::PassWrapper<SymbolTableBuildPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;
}; // struct SymbolTableBuildPass

struct MockFunctionLocalizationPass
    : public mlir::PassWrapper<MockFunctionLocalizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  auto lookupQubitId(const mlir::Value val) -> int;

  template <class CallOpTy>
  auto getCallArgIndex(CallOpTy &callOp) -> int;

  template <class CallOpTy>
  auto getMatchedOp(CallOpTy &callOp, int callArgIndex, int thisIdIndex)
      -> mlir::Operation *;

  template <class CallOpTy>
  auto getMangledName(CallOpTy &callOp) -> std::string;

  template <class CallOpTy>
  void cloneMatchedOp(CallOpTy &callOp, const std::string &newName,
                      mlir::Operation *&clonedOp, mlir::Operation *matchedOp);

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

private:
  mlir::Operation *moduleOperation;
  std::shared_ptr<mlir::OpBuilder> builder;
  std::vector<int> theseIds;
  std::list<mlir::Operation *> toWalk;
}; // struct MockFunctionLocalizationPass

} // namespace qssc::targets::mock

#endif // MOCK_FUNCTION_LOCALIZATION_H

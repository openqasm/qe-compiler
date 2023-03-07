//===- FunctionLocalization.h - Localizing function defs --------*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

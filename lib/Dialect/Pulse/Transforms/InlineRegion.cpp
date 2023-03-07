//===- InlineRegion.cpp - Inlining all dialects -----------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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
///
///  This file declares the pass for inlining all dialects
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/InlineRegion.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::pulse {

class DialectAgnosticInlinerInterface : public InlinerInterface {

  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};

void InlineRegionPass::runOnOperation() {

  auto module = getOperation();

  for (auto function : llvm::make_early_inc_range(module.getOps<FuncOp>())) {

    // Build the inliner interface.
    DialectAgnosticInlinerInterface interface(&getContext());

    for (auto caller : llvm::make_early_inc_range(function.getOps<CallOp>())) {
      auto callee =
          dyn_cast_or_null<FuncOp>(module.lookupSymbol(caller.getCallee()));
      if (!callee)
        continue;

      // Inline the functional region operation, but only clone the internal
      // region if there is more than one use.
      if (failed(inlineRegion(interface, &callee.getBody(), caller,
                              caller.getArgOperands(), caller.getResults(),
                              caller.getLoc(),
                              /*shouldCloneInlinedRegion=*/true)))
        continue;

      // If the inlining was successful then erase the call and callee if
      // possible.
      caller->dropAllDefinedValueUses();
      caller->dropAllReferences();
      caller.erase();
      if (callee.use_empty())
        callee.erase();
    }
  }
}

llvm::StringRef InlineRegionPass::getArgument() const { return "pulse-inline"; }
llvm::StringRef InlineRegionPass::getDescription() const {
  return "Inline all dialects.";
}
} // namespace mlir::pulse

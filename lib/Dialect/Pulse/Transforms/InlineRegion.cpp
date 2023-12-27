//===- InlineRegion.cpp - Inlining all dialects -----------------*- C++ -*-===//
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
///  This file declares the pass for inlining all dialects
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/InlineRegion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::pulse {

class DialectAgnosticInlinerInterface : public InlinerInterface {

  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }
};

void InlineRegionPass::runOnOperation() {

  auto module = getOperation();

  for (auto function :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {

    // Build the inliner interface.
    DialectAgnosticInlinerInterface interface(&getContext());

    for (auto caller :
         llvm::make_early_inc_range(function.getOps<func::CallOp>())) {
      auto callee = dyn_cast_or_null<mlir::func::FuncOp>(
          module.lookupSymbol(caller.getCallee()));
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

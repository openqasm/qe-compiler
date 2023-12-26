//===- SwitchOpLowering.cpp -------------------------------------*- C++ -*-===//
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
/// \file
/// This file implements patterns for lowering quir.switch
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cassert>
#include <vector>

namespace mlir::quir {

// llvm.switch i32 %flag, label switchEnd [
//     i32 caseVal_1 : label caseRegion_1
//     i32 caseVal_2 : label caseRegion_2
//   ...
// ]
// caseRegion_default:
//     // gates
//    cf.br switchEnd
// caseRegion_1:
//     // gates
//    cf.br switchEnd
// caseRegion_2:
//     // gates
//    cf.br switchEnd
// ...
// switchEnd:
// ...
struct SwitchOpLowering : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern<SwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SwitchOp switchOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
SwitchOpLowering::matchAndRewrite(SwitchOp switchOp,
                                  PatternRewriter &rewriter) const {
  auto loc = switchOp.getLoc();

  // Start by splitting the block containing the 'quir.switch' into parts.
  // The part before will contain the condition, the part after will be the
  // continuation point.
  auto *condBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  Block *continueBlock = rewriter.splitBlock(condBlock, opPosition);
  SmallVector<Value> results;
  results.reserve(switchOp.getNumResults());
  for (auto resultType : switchOp.getResultTypes())
    results.push_back(
        continueBlock->addArgument(resultType.getType(), switchOp.getLoc()));
  // Move blocks from the "default" region to the region containing
  // 'quir.switch', place it before the continuation block, and branch to it.
  auto &defaultRegion = switchOp.getDefaultRegion();
  auto *defaultBlock = &defaultRegion.front();
  Operation *defaultTerminator = defaultRegion.back().getTerminator();
  ValueRange const defaultTerminatorOperands = defaultTerminator->getOperands();
  rewriter.setInsertionPointToEnd(&defaultRegion.back());
  rewriter.create<cf::BranchOp>(loc, continueBlock, defaultTerminatorOperands);
  rewriter.eraseOp(defaultTerminator);
  rewriter.inlineRegionBefore(defaultRegion, continueBlock);

  // Move blocks from the "case" regions (if present) to the region containing
  // 'quir.switch', place it before the continuation block and branch to it. It
  // will be placed after the "default" regions.
  auto caseBlocks = std::vector<Block *>();
  auto *currBlock = continueBlock;
  auto caseOperands = std::vector<mlir::ValueRange>();
  for (auto &region : switchOp.getCaseRegions())
    if (!region.empty()) {
      caseOperands.emplace_back();
      currBlock = &region.front();
      caseBlocks.push_back(currBlock);
      Operation *caseTerminator = region.back().getTerminator();
      ValueRange const caseTerminatorOperands = caseTerminator->getOperands();
      rewriter.setInsertionPointToEnd(&region.back());
      rewriter.create<cf::BranchOp>(loc, continueBlock, caseTerminatorOperands);
      rewriter.eraseOp(caseTerminator);
      rewriter.inlineRegionBefore(region, continueBlock);
    }

  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<LLVM::SwitchOp>(
      loc, /*flag=*/switchOp.getFlag(), /*defaultDestination=*/defaultBlock,
      /*defaultOperands=*/ValueRange(),
      /*caseValues=*/switchOp.getCaseValues(), /*caseDestinations=*/caseBlocks,
      /*caseOperands=*/caseOperands);

  // Ok, we're done!
  rewriter.replaceOp(switchOp, continueBlock->getArguments());
  return success();
}

void populateSwitchOpLoweringPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  assert(context);
  patterns.add<SwitchOpLowering>(context);
}

}; // namespace mlir::quir

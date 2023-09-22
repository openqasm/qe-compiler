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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::quir {

// llvm.switch i32 %flag, label switchEnd [
//     i32 caseVal_1 : label caseRegion_1
//     i32 caseVal_2 : label caseRegion_2
//   ...
// ]
// caseRegion_default:
//     // gates
//     br switchEnd
// caseRegion_1:
//     // gates
//     br switchEnd
// caseRegion_2:
//     // gates
//     br switchEnd
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
  // auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
  // FIXME: opPosition vs Block::iterator(switchOp)??
  Block *continueBlock = rewriter.splitBlock(condBlock, opPosition);
#if 0  
  if (switchOp.getNumResults() == 0) {
    continueBlock = remainingOpsBlock;
  } else {
    continueBlock =
        rewriter.createBlock(remainingOpsBlock, switchOp.getResultTypes());
    rewriter.create<BranchOp>(loc, remainingOpsBlock);
  }
#else
  SmallVector<Value> results;
  results.reserve(switchOp.getNumResults());
  for (Type resultType : switchOp.getResultTypes())
    results.push_back(
        continueBlock->addArgument(resultType, switchOp.getLoc()));
#endif
  // Move blocks from the "default" region to the region containing
  // 'quir.switch', place it before the continuation block, and branch to it.
  auto &defaultRegion = switchOp.defaultRegion();
  auto *defaultBlock = &defaultRegion.front();
  Operation *defaultTerminator = defaultRegion.back().getTerminator();
  ValueRange defaultTerminatorOperands = defaultTerminator->getOperands();
  rewriter.setInsertionPointToEnd(&defaultRegion.back());
  rewriter.create<BranchOp>(loc, continueBlock, defaultTerminatorOperands);
  rewriter.eraseOp(defaultTerminator);
  rewriter.inlineRegionBefore(defaultRegion, continueBlock);

  // Move blocks from the "case" regions (if present) to the region containing
  // 'quir.switch', place it before the continuation block and branch to it. It
  // will be placed after the "default" regions.
  auto caseBlocks = std::vector<Block *>();
  auto *currBlock = continueBlock;
  auto caseOperands = std::vector<mlir::ValueRange>();
  for (auto &region : switchOp.caseRegions())
    if (!region.empty()) {
      caseOperands.emplace_back(ValueRange());
      currBlock = &region.front();
      caseBlocks.push_back(currBlock);
      Operation *caseTerminator = region.back().getTerminator();
      ValueRange caseTerminatorOperands = caseTerminator->getOperands();
      rewriter.setInsertionPointToEnd(&region.back());
      rewriter.create<BranchOp>(loc, continueBlock, caseTerminatorOperands);
      rewriter.eraseOp(caseTerminator);
      rewriter.inlineRegionBefore(region, continueBlock);
    }

  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<LLVM::SwitchOp>(
      loc, switchOp.flag(), /*defaultOperands=*/ValueRange(),
      /*caseOperands=*/caseOperands, switchOp.caseValues(),
      /*branchWeights=*/ElementsAttr(), defaultBlock, caseBlocks);

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

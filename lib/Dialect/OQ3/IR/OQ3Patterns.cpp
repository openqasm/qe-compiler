//===- OQ3Patterns.cpp - OpenQASM 3 Declarative Rewrite Patterns -*- C++
//-*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file defines the OpenQASM 3 declarative rewrite rules (DRR), or
/// patterns.
///
//===----------------------------------------------------------------------===//

#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/OQ3/IR/OQ3Types.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace oq3;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Dialect/OQ3/IR/OQ3Patterns.inc"
} // anonymous namespace

// this pattern is defined by the TableGen DRR in OQ3Patterns.td
void CBitNotOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<CBitNotNotPat>(context);
}

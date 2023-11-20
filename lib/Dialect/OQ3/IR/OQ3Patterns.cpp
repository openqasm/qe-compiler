//===- OQ3Patterns.cpp - OpenQASM 3 DRR Patterns ----------------*- C++ -*-===//
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
/// This file defines the OpenQASM 3 declarative rewrite rules (DRR), or
/// patterns.
///
//===----------------------------------------------------------------------===//

#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/OQ3/IR/OQ3Types.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace oq3;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Dialect/OQ3/IR/OQ3Patterns.inc"

static std::optional<mlir::Value>
getI1InputFromExtensionOp(mlir::Operation *op) {
  if (!op)
    return std::nullopt;

  if (!mlir::isa<mlir::arith::ExtUIOp>(op) && !mlir::isa<CastOp>(op))
    return std::nullopt;

  assert(op->getNumOperands() == 1 &&
         "Cannot extract I1 operand from operations that don't have exactly 1 "
         "operand");
  auto operand = op->getOperand(0);

  if (operand.getType().isSignlessInteger(1))
    return operand;

  return std::nullopt;
}

// This pattern detects DAGs that result from qasm `bit x; bool y = (x==1);`
// groupings, which would expand in QUIR as:
//   %2 = ... : i1
//   %const1 = arith.constant 1 : i32
//   %ext_result = arith.extui : i1 to i32
//   %cmp_op_result = arith.cmpi eq, %ext_result, %const1 : i32
// and it eliminates the comparison operation, yielding the equivalent of `bit
// x; bool y = bool(x);`
struct EqEqOnePat : public OpRewritePattern<mlir::arith::CmpIOp> {
  EqEqOnePat(MLIRContext *context)
      : OpRewritePattern<mlir::arith::CmpIOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(mlir::arith::CmpIOp cmpOp,
                       mlir::PatternRewriter &rewriter) const
      -> LogicalResult override {

    if (cmpOp.getPredicate() != mlir::arith::CmpIPredicate::eq ||
        !cmpOp.getRhs().getType().isSignlessInteger())
      return failure();

    // is the rhs a constant 1?
    auto rhsConstant = mlir::dyn_cast<mlir::arith::ConstantIntOp>(
        cmpOp.getRhs().getDefiningOp());

    if (!rhsConstant)
      return failure();

    auto constIntAttr = rhsConstant.getValue().dyn_cast<mlir::IntegerAttr>();

    if (!constIntAttr || !constIntAttr.getValue().isOne())
      return failure();

    // is the lhs a sign extension of an i1?
    auto extendedI1OrNone =
        getI1InputFromExtensionOp(cmpOp.getLhs().getDefiningOp());

    if (!extendedI1OrNone.has_value())
      return failure();

    rewriter.replaceOp(cmpOp, {extendedI1OrNone.getValue()});
    return success();
  } // matchAndRewrite
};  // EqEqOnePat

/// This pattern matches on casts that are from and to the same type
/// and simply removes the cast
struct CastToSameType : public OpRewritePattern<CastOp> {
  CastToSameType(MLIRContext *context)
      : OpRewritePattern<CastOp>(context, /*benefit=*/1) {}
  auto matchAndRewrite(CastOp castOp, mlir::PatternRewriter &rewriter) const
      -> LogicalResult override {
    if (castOp.arg().getType() != castOp.out().getType())
      return failure();

    rewriter.replaceOp(castOp, castOp.arg());
    return success();
  } // matchAndRewrite
};  // struct CastToSameType

/// This pattern simplifies oq3.cbit_assign_bit operations for single-cbit
/// registers to variable_assign operations. The assigned bit is first cast to a
/// !quir.cbit<1> (which is transparent) and then directly assigned.
struct AssignSingleCBitToAssignVariablePattern
    : public OpRewritePattern<oq3::CBitAssignBitOp> {
  AssignSingleCBitToAssignVariablePattern(MLIRContext *context)
      : OpRewritePattern<oq3::CBitAssignBitOp>(context, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(oq3::CBitAssignBitOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (op.cbit_width() != 1)
      return failure();

    auto *context = rewriter.getContext();
    auto castOp = rewriter.create<CastOp>(
        op.getLoc(), quir::CBitType::get(context, 1), op.assigned_bit());

    rewriter.replaceOpWithNewOp<oq3::VariableAssignOp>(
        op, op.variable_nameAttr(), castOp.getResult());

    return success();
  }
};
} // anonymous namespace

// This pattern is defined by the TableGen DRR in `OQ3Patterns.td`
void CBitNotOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<CBitNotNotPat>(context);
}

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<CastToSameType>(context);
  results.insert<EqEqOnePat>(context);
  results.insert<AssignSingleCBitToAssignVariablePattern>(context);
}

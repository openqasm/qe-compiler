//===- Patterns.cpp - QUIR Declarative Rewrite Patterns ----------*-C++ -*-===//
//
// (C) Copyright IBM 2023.
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
///  This file defines the QUIR dialect patterns in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace quir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Dialect/QUIR/IR/QUIRPatterns.inc"

static llvm::Optional<mlir::Value>
getI1InputFromExtensionOp(mlir::Operation *op) {
  if (!op)
    return llvm::None;

  if (!mlir::isa<mlir::arith::ExtUIOp>(op) &&
      !mlir::isa<mlir::quir::CastOp>(op))
    return llvm::None;

  assert(op->getNumOperands() == 1 &&
         "Cannot extract I1 operand from operations that don't have exactly 1 "
         "operand");
  auto operand = op->getOperand(0);

  if (operand.getType().isSignlessInteger(1))
    return operand;

  return llvm::None;
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

    if (!extendedI1OrNone.hasValue())
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

/// This pattern simplifies quir.assign_cbit_bit operations for single-cbit
/// registers to assign_variable operations. The assigned bit is first cast to a
/// !quir.cbit<1> (which is transparent) and then directly assigned.
struct AssignSingleCbitToAssignVariablePattern
    : public OpRewritePattern<AssignCbitBitOp> {
  AssignSingleCbitToAssignVariablePattern(MLIRContext *context)
      : OpRewritePattern<AssignCbitBitOp>(context, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(AssignCbitBitOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (op.cbit_width() != 1)
      return failure();

    auto *context = rewriter.getContext();
    auto castOp = rewriter.create<CastOp>(
        op.getLoc(), quir::CBitType::get(context, 1), op.assigned_bit());

    rewriter.replaceOpWithNewOp<VariableAssignOp>(op, op.variable_nameAttr(),
                                                  castOp.getResult());

    return success();
  }
};

} // anonymous namespace

// this pattern is defined by the TableGen DRR in QUIRPatterns.td
void Cbit_NotOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<CbitNotNotPat>(context);
}

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<CastToSameType>(context);
  results.insert<EqEqOnePat>(context);
  results.insert<AssignSingleCbitToAssignVariablePattern>(context);
}

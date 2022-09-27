//===- Patterns.cpp - QUIR Declarative Rewrite Patterns ----------*-C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
} // anonymous namespace

namespace {
// This pattern detects DAGs that result from qasm `bit x; bool y = (x==1);`
// groupings, which would expand in QUIR as:
//   %bit = quir.declare_cbit : memref<1xi1>
//   %const1 = constant 1 : i32
//   %cast_result = "quir.cast"(%bit) : (memref<1xi1>) -> i32
//   %cmp_op_result = cmpi eq, %cast_result, %const1 : i32
// and it modifies the cast_result (from int to bool) and eliminates the
// comparison operation, yielding the equivalent of `bit x; bool y = bool(x);`
struct EqEqOnePat : public OpRewritePattern<CastOp> {
  EqEqOnePat(MLIRContext *context)
      : OpRewritePattern<CastOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(CastOp castOp, mlir::PatternRewriter &rewriter) const
      -> LogicalResult override {
    OpBuilder b(castOp.getContext());
    // %cast_result = "quir.cast"(%bit) : (memref<1xi1>) -> i32
    // The pattern`(x == 1)` will cast `x` from a memref to an int for
    // comparison to an integer. Thus, we check that the argument to
    // this cast operation was a memref (i.e., declared like `bit x;`)
    // and that the result of the cast is of integer type
    Value castOpArg = castOp.arg();
    Value castResult = castOp.out();
    if (!(castOpArg.getType().isa<MemRefType>() &&
          castResult.getType().isa<IntegerType>()))
      return failure();
    for (Operation *castResultUser : castResult.getUsers()) {
      if (auto cmpIOp = dyn_cast<mlir::arith::CmpIOp>(castResultUser)) {
        mlir::arith::CmpIPredicate pred = cmpIOp.getPredicate();
        // The non-constant argument to the comparison
        Value otherVal = cmpIOp.getRhs();
        if (castResult == cmpIOp.getRhs()) {
          // Take care of (1 == x) case
          otherVal = cmpIOp.getLhs();
        }
        Operation *otherValOp = otherVal.getDefiningOp();
        if (pred == mlir::arith::CmpIPredicate::eq && otherValOp) {
          if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(otherValOp)) {
            auto constVal = constOp.getValue().dyn_cast<IntegerAttr>();
            if (constVal.getInt() != 1)
              continue;
            auto newCast = rewriter.create<CastOp>(castOp.getLoc(),
                                                   b.getI1Type(), castOpArg);
            rewriter.replaceOp(castOp, newCast.out());
            rewriter.replaceOp(cmpIOp, newCast.out());
            return success();
          } // if otherVal is defined by a ConstantOp
        }   // if comparison is 'eq' and otherVal is defined by some Op
      }     // if castResultUser is CmpIOp
    }       // for cast result users
    return failure();
  } // matchAndRewrite
};  // EqEqOnePat

/// This pattern matches on dags of the form (where ? can be any type):
/// cbit 1x?
/// op -> ?
/// store ? to 1x?
/// quir.cast 1x? -> ?
/// and removes the cast and replaces the cast result with the op result
struct CastSmallCbitAfterStorePat : public OpRewritePattern<CastOp> {
  CastSmallCbitAfterStorePat(MLIRContext *context)
      : OpRewritePattern<CastOp>(context, /*benefit=*/1) {}

  auto matchAndRewrite(CastOp castOp, mlir::PatternRewriter &rewriter) const
      -> LogicalResult override {
    auto memRefType = castOp.arg().getType().dyn_cast<MemRefType>();
    if (!memRefType)
      return failure();

    ArrayRef<int64_t> shape = memRefType.getShape();
    if (shape.size() != 1 && shape[0] != 1)
      return failure();

    // uses are iterated over in reverse order -> last use first
    bool castUseFound = false;
    auto use_iter = castOp.arg().use_begin();
    while (use_iter != castOp.arg().use_end()) {
      if (use_iter->getOwner() == castOp) {
        castUseFound = true;
        break;
      }
      ++use_iter;
    }

    if (!castUseFound)
      return failure();
    ++use_iter; // get the next most recent use
    if (use_iter == castOp.arg().use_end())
      return failure();

    auto storeOp = dyn_cast<mlir::memref::StoreOp>(use_iter->getOwner());
    if (!storeOp)
      return failure();

    rewriter.replaceOp(castOp, storeOp.value());
    return success();
  } // matchAndRewrite
};  // struct CastSmallCbitAfterStorePat

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
  results.insert<CastSmallCbitAfterStorePat>(context);
}

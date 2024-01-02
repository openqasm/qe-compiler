//===- OQ3Patterns.td - OpenQASM 3 dialect DRR ------------*-- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file declares the OpenQASM 3 declarative rewrite rules (DRR), or
/// patterns.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_PATTERNS
#define OQ3_PATTERNS

include "mlir/IR/BuiltinOps.td"
include "mlir/IR/PatternBase.td"
include "Dialect/OQ3/IR/OQ3Ops.td"

def CBitNotNotPat : Pat<
    (OQ3_CBitNotOp (OQ3_CBitNotOp $in)), (replaceWithValue $in)>;

#endif // OQ3_PATTERNS

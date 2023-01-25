//===- QUIRPatterns.td - QUIR dialect declarative rewrite rules (patterns) --*--
// tablegen -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_PATTERNS
#define QUIR_PATTERNS

include "mlir/Dialect/StandardOps/IR/Ops.td"
include "Dialect/QUIR/IR/QUIR.td"

def CBitNotNotPat : Pat<
    (QUIR_CBit_NotOp (QUIR_CBit_NotOp $in)), (replaceWithValue $in)>;

#endif // QUIR_PATTERNS

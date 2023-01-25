//===- OQ3ArithmeticOps.td - OpenQASM 3 arithmetic ops -----*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the definition file for the OpenQASM 3 dialect arithmetic
/// operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_ARITHMETIC_OPS
#define OQ3_ARITHMETIC_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class OQ3_ArithmeticUnaryOp<string mnemonic, list<Trait> traits = []> :
      OQ3_UnaryOp<mnemonic,
                  !listconcat(traits,
                              [NoSideEffect, SameOperandsAndResultType])>;

class OQ3_ArithmeticBinaryOp<string mnemonic, list<Trait> traits = []> :
      OQ3_BinaryOp<mnemonic,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])> {
    let arguments = (ins AnyClassical:$lhs, AnyClassical:$rhs);
    let results = (outs AnyClassical:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

#endif // OQ3_ARITHMETIC_OPS

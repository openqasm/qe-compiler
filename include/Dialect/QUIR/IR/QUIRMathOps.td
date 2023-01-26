//===- QUIRMathOps.td - QUIR dialect math ops --------------*- tablegen -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_MATH_OPS
#define QUIR_MATH_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"

// QUIR Math is focused on math operations for the OpenQASM/QUIR custom types, CBit,
// Angle, Duration, and Stretch.

class QUIR_BinaryOp<string mnemonic, list<Trait> traits = []> :
  QUIR_Op<mnemonic, !listconcat(traits, [NoSideEffect])>, Arguments<(ins AnyClassical:$lhs, AnyClassical:$rhs)> {
    let results = (outs AnyClassical:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

class QUIR_Binary_DurationOp<string mnemonic, list<Trait> traits = []> :
  QUIR_BinaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
    let arguments = (ins QUIR_DurationType:$lhs, QUIR_DurationType:$rhs);
    let results = (outs QUIR_DurationType:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($result)
    }];
}

def QUIR_Duration_AddOp : QUIR_Binary_DurationOp<"duration_add"> {
    let summary = "Add two durations";
    let description = [{
        The `quir.duration_add` operation takes two duration operands and returns one duration
        result, which is the sum of the two operands.
    }];
}

def QUIR_Duration_SubOp : QUIR_Binary_DurationOp<"duration_sub"> {
    let summary = "Subtract two durations";
    let description = [{
        The `quir.duration_sub` operation takes two duration operands and returns one duration
        result, which is the difference of the two operands.
    }];
}

def QUIR_Duration_MulOp : QUIR_Binary_DurationOp<"duration_mul"> {
    let summary = "Multiply two durations";
    let description = [{
        The `quir.duration_mul` operation takes two duration operands and returns one duration
        result, which is the multiplication of the two operands.
    }];
}

#endif // QUIR_MATH_OPS

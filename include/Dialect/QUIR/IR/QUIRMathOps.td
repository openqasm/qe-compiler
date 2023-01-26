//===- QUIRMathOps.td - QUIR dialect math ops --------------*- tablegen -*-===//
//
// (C) Copyright IBM 2022.
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

////// TODO: remove
class QUIR_Binary_AngleOp<string mnemonic, list<Trait> traits = []> :
  QUIR_BinaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
    let arguments = (ins AnyAngle:$lhs, AnyAngle:$rhs);
    let results = (outs AnyAngle:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($result)
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

////// TODO: remove
class QUIR_Binary_CmpOp<string mnemonic, list<Trait> traits = []> :
  QUIR_BinaryOp<mnemonic, !listconcat(traits, [SameTypeOperands])> {
    let results = (outs I1:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($lhs) `->` type($result)
    }];

    let verifier = [{
        std::vector predicates = { "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge" };

        if (std::find(predicates.begin(), predicates.end(), this->predicate()) != predicates.end())
            return success();
        else
            return emitOpError("requires predicate \"eq\", \"ne\", \"slt\", \"sle\", \"sgt\", \"sge\", \"ult\", \"ule\", \"ugt\", \"uge\"");
    }];
}

////// TODO: remove
def QUIR_Angle_AddOp : QUIR_Binary_AngleOp<"angle_add"> {
    let summary = "Add two angles";
    let description = [{
        The `quir.angle_add` operation takes two angle operands and returns one angle
        result, which is the sum of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

////// TODO: remove
def QUIR_Angle_SubOp : QUIR_Binary_AngleOp<"angle_sub"> {
    let summary = "Subtract two angles";
    let description = [{
        The `quir.angle_sub` operation takes two angle operands and returns one angle
        result, which is the difference of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

////// TODO: remove
def QUIR_Angle_MulOp : QUIR_Binary_AngleOp<"angle_mul"> {
    let summary = "Multiply two angles";
    let description = [{
        The `quir.angle_mul` operation takes two angle operands and returns one angle
        result, which is the multiplication of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

////// TODO: remove
def QUIR_Angle_DivOp : QUIR_Binary_AngleOp<"angle_div"> {
    let summary = "Divide two angles";
    let description = [{
        The `quir.angle_div` operation takes two angle operands and returns one angle
        result, which is the division of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

////// TODO: remove
def QUIR_Angle_CmpOp : QUIR_Binary_CmpOp<"angle_cmp"> {
    let summary = "Compare two angles";
    let description = [{
        The `quir.angle_cmp` operation takes two angle operands and returns one boolean
        result. All angle operations are performed over the interval [0,2*pi).
    }];

	let arguments = (ins StrAttr:$predicate, AnyAngle:$lhs, AnyAngle:$rhs);
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

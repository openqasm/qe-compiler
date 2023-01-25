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

class QUIR_UnaryOp<string mnemonic, list<Trait> traits = []> :
  QUIR_Op<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let results = (outs AnyType);
}

class QUIR_UnaryOpSameOperandAndResultType<string mnemonic,
                                            list<Trait> traits = []> :
  QUIR_UnaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
}

class QUIR_Unary_CBitOp<string mnemonic, list<Trait> traits = []> :
  QUIR_UnaryOpSameOperandAndResultType<mnemonic, traits> {
    let arguments = (ins AnyCBit:$operand);
    let results = (outs AnyCBit:$result);
    let assemblyFormat = [{
        attr-dict $operand `:` type($result)
    }];
}

class QUIR_BinaryOp<string mnemonic, list<Trait> traits = []> :
  QUIR_Op<mnemonic, !listconcat(traits, [NoSideEffect])>, Arguments<(ins AnyClassical:$lhs, AnyClassical:$rhs)> {
    let results = (outs AnyClassical:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

class QUIR_Binary_CBitOp<string mnemonic, list<Trait> traits = []> :
  QUIR_BinaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
    let arguments = (ins AnyCBit:$lhs, AnyCBit:$rhs);
    let results = (outs AnyCBit:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($result)
    }];
}

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

// def QUIR_CBit_NotOp : QUIR_Unary_CBitOp<"cbit_not"> {
//     let summary = "cbit and register bit-wise negation";
//     let description = [{
//         The `oq3.cbit_not` operation takes one cbit operand and returns one cbit
//         result, which is the bit-wise negation of the operand.
//     }];

//     let hasCanonicalizer = 1;
// }

// def QUIR_CBit_RotLOp : QUIR_BinaryOp<"cbit_rotl"> {
//     let summary = "cbit and register barrel-shift left";
//     let description = [{
//         The `quir.cbit_rotl` operation takes one cbit operand and one unsigned operand
//         and returns one cbit result, which is the bit-wise barrel-shift left by the
//         amount specified.
//     }];

//     let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
// }

// def QUIR_CBit_RotROp : QUIR_BinaryOp<"cbit_rotr"> {
//     let summary = "cbit and register barrel-shift right";
//     let description = [{
//         The `quir.cbit_rotr` operation takes one cbit operand and one unsigned operand
//         and returns one cbit result, which is the bit-wise barrel-shift right by the
//         amount specified.
//     }];

//     let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
// }

// def QUIR_CBit_PopcountOp : QUIR_UnaryOp<"cbit_popcount"> {
//     let summary = "cbit and register bit-wise population count";
//     let description = [{
//         The `quir.cbit_popcount` operation takes one cbit operand and returns one cbit
//         result, which is the bit-wise negation of the operand.
//     }];

//     let arguments = (ins AnyCBit:$operand);
//     let results = (outs AnyInteger:$result);
//     let assemblyFormat = [{
//         attr-dict $operand `:` `(` type($operand) `)` `->` type($result)
//     }];
// }

// def QUIR_CBit_AndOp : QUIR_Binary_CBitOp<"cbit_and"> {
//     let summary = "cbit and register bit-wise and";
//     let description = [{
//         The `quir.cbit_and` operation takes two cbit operands and returns one cbit
//         result, which is the bit-wise and of the operands.
//     }];
// }

// def QUIR_CBit_OrOp : QUIR_Binary_CBitOp<"cbit_or"> {
//     let summary = "cbit and register bit-wise or";
//     let description = [{
//         The `quir.cbit_or` operation takes two cbit operands and returns one cbit
//         result, which is the bit-wise or of the operands.
//     }];
// }

// def QUIR_CBit_XorOp : QUIR_Binary_CBitOp<"cbit_xor"> {
//     let summary = "cbit and register bit-wise xor";
//     let description = [{
//         The `quir.cbit_xor` operation takes two cbit operands and returns one cbit
//         result, which is the bit-wise xor of the operands.
//     }];
// }

// def QUIR_CBit_RshiftOp : QUIR_BinaryOp<"cbit_rshift"> {
//     let summary = "cbit and register logical shift right";
//     let description = [{
//         The `quir.cbit_rshift` operation takes one cbit operand and one unsigned operand
//         and returns one cbit result, which is the bit-wise logical-shift right by the
//         amount specified.
//     }];

//     let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
// }

// def QUIR_CBit_LshiftOp : QUIR_BinaryOp<"cbit_lshift"> {
//     let summary = "cbit and register logical shift left";
//     let description = [{
//         The `quir.cbit_lshift` operation takes one cbit operand and one unsigned operand
//         and returns one cbit result, which is the bit-wise logical-shift left by the
//         amount specified.
//     }];

//     let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
// }

// def QUIR_CBit_ExtractBitOp : QUIR_Op<"cbit_extractbit", [NoSideEffect]> {
//     let summary = "extract single bit from a bitmap";
//     let description = [{
//         The `quir.cbit_extractbit` operations takes a bitmap operand (cbit or
//         integer) and one index attribute and returns one i1 result, which is
//         the bit at the position with the given index.

//         Example:

//         ```mlir
//         %6 = quir.cbit_extractbit(%5 : !quir.cbit<2>) [1] : i1

//         ```
//     }];

//     let arguments = (ins AnyCBitOrSignlessInteger:$operand, IndexAttr:$index);

//     let results = (outs I1:$result);

//     let assemblyFormat = [{
//         attr-dict `(` $operand `:` type($operand) `)` `[` $index `]` `:` type($result)
//     }];

//     let hasFolder = 1;
// }

// def QUIR_CBit_InsertBitOp : QUIR_Op<"cbit_insertbit", [
//             NoSideEffect,
//             TypesMatchWith<
//                 "first operand and result have the same type",
//                 "operand", "result", "$_self">]> {
//     let summary = "insert single bit into a bitmap";
//     let description = [{
//         The `quir.cbit_insertbit` operations takes a bitmap operand (cbit or
//         integer), a i1 operand, and one index attribute and returns the bitmap
//         with the bit at the position with the given index replaced by the
//         provided bit operand.

//         Example:

//         ```mlir
//         %6 = quir.cbit_insertbit(%5 : !quir.cbit<2>) [1] = %4 : !quir.cbit<2>

//         ```
//     }];

//     let arguments = (ins AnyCBitOrSignlessInteger:$operand, I1:$assigned_bit, IndexAttr:$index);

//     let results = (outs AnyCBitOrSignlessInteger:$result);

//     let assemblyFormat = [{
//         attr-dict `(` $operand `:` type($operand) `)` `[` $index `]` `=` $assigned_bit `:` type($result)
//     }];
// }



def QUIR_Angle_AddOp : QUIR_Binary_AngleOp<"angle_add"> {
    let summary = "Add two angles";
    let description = [{
        The `quir.angle_add` operation takes two angle operands and returns one angle
        result, which is the sum of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

def QUIR_Angle_SubOp : QUIR_Binary_AngleOp<"angle_sub"> {
    let summary = "Subtract two angles";
    let description = [{
        The `quir.angle_sub` operation takes two angle operands and returns one angle
        result, which is the difference of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

def QUIR_Angle_MulOp : QUIR_Binary_AngleOp<"angle_mul"> {
    let summary = "Multiply two angles";
    let description = [{
        The `quir.angle_mul` operation takes two angle operands and returns one angle
        result, which is the multiplication of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

def QUIR_Angle_DivOp : QUIR_Binary_AngleOp<"angle_div"> {
    let summary = "Divide two angles";
    let description = [{
        The `quir.angle_div` operation takes two angle operands and returns one angle
        result, which is the division of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

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

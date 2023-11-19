//===- OQ3CBitOps.td - OpenQASM 3 classical bit ops -*- tablegen --------*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the operation definition file for OpenQASM 3 operations on
/// classical bits.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_CBIT_OPS
#define OQ3_CBIT_OPS

include "Dialect/OQ3/IR/OQ3Base.td"

class OQ3_CBitUnaryOp<string mnemonic, list<Trait> traits = []> :
  OQ3_ArithmeticUnaryOp<mnemonic, traits> {
    let arguments = (ins AnyCBit:$operand);
    let results = (outs AnyCBit:$result);
    let assemblyFormat = [{
        attr-dict $operand `:` type($result)
    }];
}

class OQ3_CBitBinaryOp<string mnemonic, list<Trait> traits = []> :
  OQ3_BinaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
    let arguments = (ins AnyCBit:$lhs, AnyCBit:$rhs);
    let results = (outs AnyCBit:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($result)
    }];
}

// -----

def OQ3_CBitAssignBitOp : OQ3_Op<"cbit_assign_bit",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Assign a single bit in a classical bit register";
    let description = [{
        The operation `oq3.cbit_assign_bit` assigns a new value to an
        individual bit of a classical bit register in a variable.

        Example:

        ```mlir
        oq3.cbit_assign_bit "b"<2> [0] : i1 = %9
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        IndexAttr:$index,
        IndexAttr:$cbit_width,
        I1:$assigned_bit
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $variable_name `<` $cbit_width `>` `[` $index `]` `:` type($assigned_bit) `=` $assigned_bit
    }];
}

// -----

def OQ3_CBitNotOp : OQ3_CBitUnaryOp<"cbit_not"> {
    let summary = "cbit and register bit-wise negation";
    let description = [{
        The `oq3.cbit_not` operation takes one cbit operand and returns one cbit
        result, which is the bit-wise negation of the operand.
    }];

    let hasCanonicalizer = 1;
}

// -----

def OQ3_CBitRotLOp : OQ3_BinaryOp<"cbit_rotl"> {
    let summary = "cbit and register barrel-shift left";
    let description = [{
        The `oq3.cbit_rotl` operation takes one cbit operand and one unsigned operand
        and returns one cbit result, which is the bit-wise barrel-shift left by the
        amount specified.
    }];

    let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
}

// -----

def OQ3_CBitRotROp : OQ3_BinaryOp<"cbit_rotr"> {
    let summary = "cbit and register barrel-shift right";
    let description = [{
        The `oq3.cbit_rotr` operation takes one cbit operand and one unsigned operand
        and returns one cbit result, which is the bit-wise barrel-shift right by the
        amount specified.
    }];

    let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
}

// -----

def OQ3_CBitPopcountOp : OQ3_UnaryOp<"cbit_popcount"> {
    let summary = "cbit and register bit-wise population count";
    let description = [{
        The `oq3.cbit_popcount` operation takes one cbit operand and returns one cbit
        result, which is the bit-wise negation of the operand.
    }];

    let arguments = (ins AnyCBit:$operand);
    let results = (outs AnyInteger:$result);
    let assemblyFormat = [{
        attr-dict $operand `:` `(` type($operand) `)` `->` type($result)
    }];
}

// -----

def OQ3_CBitAndOp : OQ3_CBitBinaryOp<"cbit_and"> {
    let summary = "cbit and register bit-wise and";
    let description = [{
        The `oq3.cbit_and` operation takes two cbit operands and returns one cbit
        result, which is the bit-wise and of the operands.
    }];
}

// -----

def OQ3_CBitOrOp : OQ3_CBitBinaryOp<"cbit_or"> {
    let summary = "cbit and register bit-wise or";
    let description = [{
        The `oq3.cbit_or` operation takes two cbit operands and returns one cbit
        result, which is the bit-wise or of the operands.
    }];
}

// -----

def OQ3_CBitXorOp : OQ3_CBitBinaryOp<"cbit_xor"> {
    let summary = "cbit and register bit-wise xor";
    let description = [{
        The `oq3.cbit_xor` operation takes two cbit operands and returns one cbit
        result, which is the bit-wise xor of the operands.
    }];
}

// -----

def OQ3_CBitRShiftOp : OQ3_BinaryOp<"cbit_rshift"> {
    let summary = "cbit and register logical shift right";
    let description = [{
        The `oq3.cbit_rshift` operation takes one cbit operand and one unsigned operand
        and returns one cbit result, which is the bit-wise logical-shift right by the
        amount specified.
    }];

    let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
}

// -----

def OQ3_CBitLShiftOp : OQ3_BinaryOp<"cbit_lshift"> {
    let summary = "cbit and register logical shift left";
    let description = [{
        The `oq3.cbit_lshift` operation takes one cbit operand and one unsigned operand
        and returns one cbit result, which is the bit-wise logical-shift left by the
        amount specified.
    }];

    let arguments = (ins AnyCBit:$lhs, AnyInteger:$rhs);
}

// -----

def OQ3_CBitExtractBitOp : OQ3_Op<"cbit_extractbit", [Pure]> {
    let summary = "extract single bit from a bitmap";
    let description = [{
        The `oq3.cbit_extractbit` operations takes a bitmap operand (cbit or
        integer) and one index attribute and returns one i1 result, which is
        the bit at the position with the given index.

        Example:

        ```mlir
        %6 = oq3.cbit_extractbit(%5 : !oq3.cbit<2>) [1] : i1

        ```
    }];

    let arguments = (ins AnyCBitOrSignlessInteger:$operand, IndexAttr:$index);

    let results = (outs I1:$result);

    let assemblyFormat = [{
        attr-dict `(` $operand `:` type($operand) `)` `[` $index `]` `:` type($result)
    }];

    let hasFolder = 1;
}

// -----

def OQ3_CBitInsertBitOp : OQ3_Op<"cbit_insertbit", [
            Pure,
            TypesMatchWith<
                "first operand and result have the same type",
                "operand", "result", "$_self">]> {
    let summary = "insert single bit into a bitmap";
    let description = [{
        The `oq3.cbit_insertbit` operations takes a bitmap operand (cbit or
        integer), a i1 operand, and one index attribute and returns the bitmap
        with the bit at the position with the given index replaced by the
        provided bit operand.

        Example:

        ```mlir
        %6 = oq3.cbit_insertbit(%5 : !oq3.cbit<2>) [1] = %4 : !quir.cbit<2>

        ```
    }];

    let arguments = (ins AnyCBitOrSignlessInteger:$operand, I1:$assigned_bit, IndexAttr:$index);

    let results = (outs AnyCBitOrSignlessInteger:$result);

    let assemblyFormat = [{
        attr-dict `(` $operand `:` type($operand) `)` `[` $index `]` `=` $assigned_bit `:` type($result)
    }];
}

#endif // OQ3_CBIT_OPS

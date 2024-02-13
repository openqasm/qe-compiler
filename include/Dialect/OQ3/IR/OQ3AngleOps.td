//===- OQ3AngleOps.td - OpenQASM 3 angle ops --------*- tablegen --------*-===//
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
/// angles.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_ANGLE_OPS
#define OQ3_ANGLE_OPS

include "Dialect/OQ3/IR/OQ3Base.td"

class OQ3_BinaryAngleOp<string mnemonic, list<Trait> traits = []> :
  OQ3_BinaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
    let arguments = (ins AnyAngle:$lhs, AnyAngle:$rhs);
    let results = (outs AnyAngle:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($result)
    }];
}

class OQ3_BinaryCmpOp<string mnemonic, list<Trait> traits = []> :
  OQ3_BinaryOp<mnemonic, !listconcat(traits, [SameTypeOperands])> {
    let results = (outs I1:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($lhs) `->` type($result)
    }];

    let hasVerifier = 1;
}

// -----

def OQ3_AngleAddOp : OQ3_BinaryAngleOp<"angle_add"> {
    let summary = "Add two angles";
    let description = [{
        The `oq3.angle_add` operation takes two angle operands and returns one angle
        result, which is the sum of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

// -----

def OQ3_AngleSubOp : OQ3_BinaryAngleOp<"angle_sub"> {
    let summary = "Subtract two angles";
    let description = [{
        The `oq3.angle_sub` operation takes two angle operands and returns one angle
        result, which is the difference of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

// -----

def OQ3_AngleMulOp : OQ3_BinaryAngleOp<"angle_mul"> {
    let summary = "Multiply two angles";
    let description = [{
        The `oq3.angle_mul` operation takes two angle operands and returns one angle
        result, which is the multiplication of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

// -----

def OQ3_AngleDivOp : OQ3_BinaryAngleOp<"angle_div"> {
    let summary = "Divide two angles";
    let description = [{
        The `oq3.angle_div` operation takes two angle operands and returns one angle
        result, which is the division of the two operands. All angle operations are performed
        over the interval [0,2*pi).
    }];
}

// -----

def OQ3_AngleCmpOp : OQ3_BinaryCmpOp<"angle_cmp"> {
    let summary = "Compare two angles";
    let description = [{
        The `oq3.angle_cmp` operation takes two angle operands and returns one boolean
        result. All angle operations are performed over the interval [0,2*pi).
    }];

	let arguments = (ins StrAttr:$predicate, AnyAngle:$lhs, AnyAngle:$rhs);
}

#endif // OQ3_ANGLE_OPS

//===- OQ3GateOps.td - OpenQASM 3 dialect gate ops ---------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the operation definition file for OpenQASM 3 gate operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_GATE_OPS
#define OQ3_GATE_OPS

//===----------------------------------------------------------------------===//
// Duration ops
//===----------------------------------------------------------------------===//
class OQ3_BinaryDurationOp<string mnemonic, list<Trait> traits = []> :
  OQ3_BinaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
    let arguments = (ins AnyDuration:$lhs, AnyDuration:$rhs);
    let results = (outs AnyDuration:$result);
    let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` type($result)
    }];
}

// -----

def OQ3_DeclareDurationOp : OQ3_Op<"declare_duration", [NoSideEffect]> {
    let summary = "Declare a new duration.";
    let description = [{
        The `oq3.declare_duration` operation creates a new OpenQASM 3 duration, representing a duration of time.

        Example:
        ```mlir
        %dur = oq3.declare_duration {value = "10ns"} : !oq3.duration
        ```
    }];

    let arguments = (ins StrAttr:$value);
    let results = (outs AnyDuration:$out);

    let assemblyFormat = [{
        attr-dict `:` type($out)
    }];
}

// -----

def OQ3_DurationAddOp : OQ3_BinaryDurationOp<"duration_add"> {
    let summary = "Add two durations";
    let description = [{
        The `oq3.duration_add` operation takes two duration operands and returns one duration
        result, which is the sum of the two operands.
    }];
}

// -----

def OQ3_DurationSubOp : OQ3_BinaryDurationOp<"duration_sub"> {
    let summary = "Subtract two durations";
    let description = [{
        The `oq3.duration_sub` operation takes two duration operands and returns one duration
        result, which is the difference of the two operands.
    }];
}

// -----

def OQ3_DurationMulOp : OQ3_BinaryDurationOp<"duration_mul"> {
    let summary = "Multiply two durations";
    let description = [{
        The `oq3.duration_mul` operation takes two duration operands and returns one duration
        result, which is the multiplication of the two operands.
    }];
}

//===----------------------------------------------------------------------===//
// Stretch ops
//===----------------------------------------------------------------------===//
def OQ3_DeclareStretchOp : OQ3_Op<"declare_stretch", [NoSideEffect]> {
    let summary = "Declare a new stretch.";
    let description = [{
        The `oq3.declare_stretch` operation creates a new QUIR `Stretch` type, representing an unknown duration of time.

        Example:
        ```mlir
        %dur = "oq3.declare_stretch"() : () -> !quir.stretch
        ```
    }];

    let results = (outs AnyStretch:$out);

    let assemblyFormat = [{
        attr-dict `:` type($out)
    }];
}

#endif // OQ3_GATE_OPS

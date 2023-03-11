//===- OQ3CastOps.td - OpenQASM 3 dialect cast ops ---------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the operation definition file for OpenQASM 3 cast operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_CAST_OPS
#define OQ3_CAST_OPS

def OQ3_CastOp : OQ3_Op<"cast", [NoSideEffect]> {
    let summary = "Cast between classical types";
    let description = [{
        The `oq3.cast` operation represents a cast between different classical types, including non-OQ3 types.

        Example:
        ```mlir
        %ang1 = "oq3.cast"(%creg) : (memref<10xi1>) -> !quir.angle<20>
        ```
    }];

    let arguments = (ins AnyClassical:$arg);
    let results = (outs AnyClassical:$out);

    let hasCanonicalizer = 1;
}

#endif // OQ3_CAST_OPS

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

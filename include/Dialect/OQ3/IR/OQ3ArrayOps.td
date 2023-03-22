//===- OQ3ArrayOps.td - OpenQASM 3 dialect array ops -------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the operation definition file for OpenQASM 3 array operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_ARRAY_OPS
#define OQ3_ARRAY_OPS

include "mlir/IR/SymbolInterfaces.td"

def OQ3_DeclareArrayOp : OQ3_Op<"declare_array", [Symbol]> {
    let summary = "Declares a classical array";
    let description = [{
        The `oq3.declare_array` operation declares a classical array
        with the given name (sym_name), element type, and number of elements.

        Example:
        ```mlir
        oq3.declare_array "myVar" : i1[2]
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        IndexAttr:$num_elements
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $sym_name `:` $type `[` $num_elements `]`
    }];

    let verifier = [{
        auto t = (*this).type();

        if( t.isa<::mlir::quir::AngleType>() || t.isa<::mlir::quir::CBitType>() || t.isa<::mlir::quir::DurationType>() || t.isa<::mlir::quir::StretchType>() || t.isIntOrIndexOrFloat())
            return success();
        return failure();
    }];
}

// -----

def OQ3_AssignArrayElementOp : OQ3_Op<"assign_array_element",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Assigns a new value to an element of a classical array";
    let description = [{
        The operation `oq3.assign_array_element` assigns a new value to an
        element of a classical array, where the array is specified by name and
        the element by index.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        IndexAttr:$index,
        AnyClassical:$assigned_value
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $variable_name `[` $index `]` `:` type($assigned_value) `=` $assigned_value
    }];
    // op is verified by its traits
    let verifier = ?;
}

// -----

def OQ3_UseArrayElementOp : OQ3_Op<"use_array_element",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Use the current value of an array element";
    let description = [{
        The operation `oq3.use_array_element` returns the current value (wrt
        to `oq3.assign_array_element` operations) of an element in a classical
        array, where the array is specified by name and the element by index.

        Example:

        ```mlir
        %2 = oq3.use_array_element "result"[6] : i1
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        IndexAttr:$index
    );

    let results = (outs AnyClassical:$res);

    let assemblyFormat = [{
        $variable_name `[` $index `]` `:` type($res) attr-dict
    }];
    // op is verified by its traits
    let verifier = ?;
}

#endif // OQ3_ARRAY_OPS

//===- OQ3VariableOps.td - OpenQASM 3 dialect variable ops -*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the operation definition file for OpenQASM 3 variable-related
/// operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_VARIABLE_OPS
#define OQ3_VARIABLE_OPS

def OQ3_DeclareVariableOp : OQ3_Op<"declare_variable", [Symbol]> {
    let summary = "Declares a classical variable";
    let description = [{
        The `oq3.declare_variable` operation declares a classical variable
        with the given name (sym_name), type, and an optional constant
        initializer.
        If present, the attributes `input` and `output` indicate that this
        variable is an input or output variable, respectively.

        Example:

        ```mlir
        oq3.declare_variable "myVar" : i1 = true
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        UnitAttr:$input,
        UnitAttr:$output,
        OptionalAttr<AnyAttr>:$initial_value
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $sym_name `:` $type (`=` $initial_value^)?
    }];

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$sym_name, "::mlir::TypeAttr":$type), [{
            $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
            $_state.addAttribute("type", type);
        }]>
    ];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        bool isInputVariable() { return getInput(); }
        bool isOutputVariable() { return getOutput(); }
    }];
}

// -----

def OQ3_VariableAssignOp : OQ3_Op<"variable_assign",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Assign a new value to a classical variable";
    let description = [{
        The operation `oq3.variable_assign` assigns a new value to a classical
        variable given by name.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        AnyClassical:$assigned_value
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $variable_name `:` type($assigned_value) `=` $assigned_value
    }];
}

// -----

def OQ3_VariableLoadOp : OQ3_Op<"variable_load",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Use the current value of a variable";
    let description = [{
        The operation `oq3.variable_load` returns the current value (wrt to
        `oq3.variable_assign` operations) of the classical variable with the
        given name.

        Example:

        ```mlir
        %2 = oq3.variable_load "a" : !quir.cbit<1>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name
    );

    let results = (outs AnyClassical:$res);

    let assemblyFormat = [{
        $variable_name `:` type($res) attr-dict
    }];
}

#endif // OQ3_VARIABLE_OPS

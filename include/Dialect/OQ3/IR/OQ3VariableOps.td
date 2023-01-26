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

    let verifier = [{
        auto t = (*this).type();

        if( t.isa<::mlir::quir::AngleType>() || t.isa<::mlir::quir::CBitType>() || t.isa<::mlir::quir::DurationType>() || t.isa<::mlir::quir::StretchType>() || t.isIntOrIndexOrFloat() || t.isa<ComplexType>())
            return success();
        std::string str;
        llvm::raw_string_ostream os(str);
        t.print(os);

        return emitOpError("MLIR type " + str + " not supported for declarations.");
    }];

    let extraClassDeclaration = [{
        bool isInputVariable() { return input(); }
        bool isOutputVariable() { return output(); }
    }];
}

// -----

def OQ3_AssignVariableOp : OQ3_Op<"assign_variable",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Assign a new value to a classical variable";
    let description = [{
        The operation `oq3.assign_variable` assigns a new value to a classical
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
    // op is verified by its traits
    let verifier = ?;
}

// -----

def OQ3_UseVariableOp : OQ3_Op<"use_variable",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Use the current value of a variable";
    let description = [{
        The operation `oq3.use_variable` returns the current value (wrt to
        `oq3.assign_variable` operations) of the classical variable with the
        given name.

        Example:

        ```mlir
        %2 = oq3.use_variable "a" : !quir.cbit<1>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name
    );

    let results = (outs AnyClassical:$res);

    let assemblyFormat = [{
        $variable_name `:` type($res) attr-dict
    }];
    // op is verified by its traits
    let verifier = ?;
}

#endif // OQ3_VARIABLE_OPS
